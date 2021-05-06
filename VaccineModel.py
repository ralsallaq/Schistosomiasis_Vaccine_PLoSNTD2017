import sys, os, time
import numpy as np
from scipy.misc import comb
from scipy import linalg
from scipy import integrate # odeint is part of the submodule integrate
from scipy.sparse import spdiags
#import pylab as pl
#import warnings
#demography module
import ObtainDemography_optimized
reload(ObtainDemography_optimized)
ObtainDemographic= ObtainDemography_optimized.ObtainDemographic

#warnings.filterwarnings("ignore")
"""
This is a Schistomiasis vaccine model built in python with the following features:
1. Fully automated, it only takes a file containing age groups and their parameters in their respective places.
2. Structured worm burden (SWB) with the ability to automatically create any number of SWBs: stratified by age and vaccine-status
3. Tracking of the vaccinated and the unvaccinated populations over years ==>important for many analyses such as the cost-effectiveness analyses
4. Incorporates snail FOI as =Bee*TotalEggsReleasedByHumans/HumanNumber

Created by Ramzi Alsallaq for the Schistomiasis modeling project PIs: Drs Charles King and David Gurarie

October 30, 2015
"""


class SWBPOP:
    """Defines the SWBPOP"""
    def __init__(self, **kwargs):
        # worm distribution params
        self.p={}
        #demography:
        self.p['Fpop'] = 0.25  #fraction in SWB from the whole population
        self.p['Mortrate'] = 0.05
        self.p['Matrate'] = 0.2
        self.p['mu'] = self.p['Mortrate'] + self.p['Matrate']  #turnover = maturation rate + mortality rate
       # self.p['Sus_source'] = self.p['mu']
        #within host:
        self.p['gamma']= 1.0
        self.p['deldblu'] = 10.
        self.p['Nstrata']= 26 #~int(self.p['lamb']/self.p['gamma'])*2
        # egg release and transmission params
        # mean egg release by h-w strata rho=rho0*exp(-k/k0)
        self.p['rho0']= 20.
        self.p['kay0']= 9.
        # Negative binomial random generator for eggs released--clumping param
        self.p['clpara']= 0.05
        #The underlying transmission coefficient from snails to humans
        self.p['Aay']=1.6
        self.p['Bee']=1.6*0.05
        #vaccine related make sure to initialize to unvaccinated status
        self.p['Vwanning']= 100. #How long it last =1./Vwanning
        self.p['VEff_fecund']=0.0
        self.p['VEff_sus']=0.0
        self.p['VEff_wmort']=0.0

        self.nn = np.arange( self.p['Nstrata'] )
        self.Sources= np.zeros(self.p['Nstrata'])
        self.State= np.zeros(self.p['Nstrata'])
        #self.State= np.hstack((self.p['Fpop'], np.zeros(self.p['Nstrata']-1)))  
 
        # initialize according to kwargs
        self.p.update(**kwargs)

    def lamb(self, zstate):
        """ if zstate is provided as a single number
            then the instantaneous lamb is returned
            otherwise it is time dependent"""
        return self.p['Aay']*zstate

    def phi(self):
        """ mean number of mated female worms given mean worm burden """
        dblu = self.p['deldblu']*self.nn
        #temp = np.array(map(int,(dblu/2.)))
        temp = np.floor(dblu/2.)
        calc_comb = comb(dblu, temp)
        calc_comb[calc_comb==np.inf]=sys.float_info.max
        return temp*(1.-np.power(2,-dblu)*calc_comb)

    def rho(self):
        """ Worm fecundity = mean number of eggs per mated female worm """
        #return self.p['rho0']*np.ones((self.nn).shape)
        return self.p['rho0']*np.exp(-self.nn/self.p['kay0'])
    def insTotalWormBurden(self):
        """ worm burden at a time point = sum_k{=0 to n}(dw*h_k*k) """
        return np.sum(self.p['deldblu']*self.nn*self.State)

    def insEggsReleased(self):
        """ instantaneous total egg release from SWB
            This is defined by the sum_k{k=1,..,n} of rho_k*phi_k*h_k """
        return np.sum(self.phi()[1:]*self.rho()[1:]*self.State[1:])

    #time dependent properties
    def getStates(self, times, States):
        self.times = times
        self.States = States
        self.TotalNumberExceptLast = (self.States[:,:-1]).sum(axis=1)
        self.TotalNumber = (self.States).sum(axis=1)
        self.NormStates = self.States/self.TotalNumber[:,None]  #there is broadcasting here
        dblu1up = self.p['deldblu']*self.nn[1:]
        ProbofCouples = 1.-2./np.power(2,dblu1up) #excluding when all F or all M
        self.Prev =  (self.States[:,1:]*ProbofCouples[None,:]).sum(axis=1)/self.TotalNumber
    def TotalWormBurden(self):
        return (self.nn*self.p['deldblu']*self.States).sum(axis=1)  #simple broadcasting
    def EggsReleased(self): #total egg release from SWB
        """ This is defined by the sum_k{k=1,..,n} of rho_k*phi_k*h_k """
        return (self.phi()[1:]*self.rho()[1:]*self.States[:,1:]).sum(axis=1)


#@profile
def dState(State, time, Sources,  Amatrix_fixed,  Aay_, Bee_, gamma_, rho0_, kay0_, phi_k, Nstrata, deldblu, Nage, Nvaccine, sigma, epsi,  Vwanning, VaccineTimes, Events, VEff_fecund, VEff_sus, VEff_wmort, TE_wmort):
    """ This function takes A,x,b; where x=state variables, A is the matrix of coefficient for the algebric system, and b is
         the source term; and returns Ax+b as the right hand side for the dx/dt at t=t0
        Amatrix here is Nvaccine*Nage by 1 array of objects, each object is Nstrata by Nstrata*Nvaccine*Nage matrix  """
# Aay = the rate at which unit density of infected snails in neighboring waters would susccessfully result in establishing one unit of worm burden (deldblu) in human population
# deldblu = threshold of worm burden increment
# Bee = the rate at which one released egg by a human host would establish infection in snails 
# sigma = the rate at which infected snails become patent
# epsi = the rate at which patent snails die

    """ allhstrata = (k=worm, a=age, j=vaccine) =n*a*j (e.g. = 25*4*2=200) """
    flattened_size = Nstrata*Nage*Nvaccine
    hstrata = State[xrange(flattened_size)] #humanstrata
    ystate = State[xrange(flattened_size,flattened_size+2)]

    output = np.zeros(hstrata.shape[0]+ystate.shape[0])
    lamb_vector = np.zeros(Nvaccine*Nage)
    Lambda_vector = np.zeros(Nvaccine*Nage)
    strata_num = np.arange(Nstrata)

    """ Parameters at this time instance """
    #when TE_m is not zero treatment is ON for the duration of the efficacy TE_m and it is applied in combination with vaccine
    VRate = np.zeros(Nage) 
    Vwanning_t = Vwanning*np.ones(Nage)
    if Events is not None:
        #for each age treated/vaccinated provide ageindex and corresponding coverage function
        GVaccinated = Events['GVaccinated']
        VRate[GVaccinated] = Events['VRate'](time-VaccineTimes) #VRate for each group, when zero no vaccination or treatment

    """ define the time varying rates of vaccination and wanning """
    TreatmentDur = 4.*7./365. # as measured by the cure rate over 4 weeks as per Cioli et al http://www.sciencedirect.com/science/article/pii/S0166685114000759 
        
#    gamma_vector = np.append(gamma_, (1.+VEff_wmort)*gamma_*(1.+TE_wmort)) #vaccine + treatment if all the efficacies are not zero for the vaccinated category
    gamma_vector = np.append(gamma_*(1.+TE_wmort), (1.+VEff_wmort)*gamma_*(1.+TE_wmort)) #vaccine + mass treatment if TE_m not zero
#    gamma_vector = np.append(gamma_, (1.+VEff_wmort)*gamma_) #vaccine only 

    #lambda = A*z*Ns/dw when h0-->h1 (dw step)
    lamb_vector[0:Nage] = Aay_*ystate[1]
    lamb_vector[Nage:Nage*Nvaccine] =(1.-VEff_sus)*Aay_*ystate[1]
    for j in xrange(Nage):
        Lambda_vector[j] = Bee_[j]*np.sum(phi_k[1:]*rho0_[j]*np.exp(-strata_num[1:]/kay0_[j])*hstrata[j*Nstrata+1:j*Nstrata+Nstrata])
    for j in xrange(Nage,Nage*Nvaccine):
        Lambda_vector[j] = (1.-VEff_fecund)*Bee_[j-Nage]*np.sum(phi_k[1:]*rho0_[j-Nage]*np.exp(-strata_num[1:]/kay0_[j-Nage])*hstrata[j*Nstrata+1:j*Nstrata+Nstrata])

    """ ----------------- Updating Amatrix by updating the time dependent parts -------------------"""

    # make a copy of the fixed matrix, so that it stays intact **important not to change the fixed part
    Amatrix = Amatrix_fixed.copy()
    """ ------ Update Amatrix with the time varying elements ----"""
    """ Unvaccinated """
    for i in xrange(0, Nage*Nstrata, Nstrata):
        np.fill_diagonal(Amatrix[i+1:i+Nstrata, i:i+Nstrata-1],  lamb_vector[i/Nstrata]) #lower diagonal
        np.fill_diagonal(Amatrix[i:i+Nstrata, i:i+Nstrata], np.diag(Amatrix[i:i+Nstrata, i:i+Nstrata])-lamb_vector[i/Nstrata] -VRate[i/Nstrata]-np.asarray([float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]))
        np.fill_diagonal(Amatrix[i:i+Nstrata-1, i+1:i+Nstrata], np.asarray([float(j+1)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)])) #upper diagonal 
        Amatrix[i:i+Nstrata,i+Nage*Nstrata:i+Nage*Nstrata+Nstrata] =Vwanning_t[i/Nstrata]*np.identity(Nstrata) #vaccine updates

    """ Vaccinated """
    for i in xrange(Nage*Nstrata, Nage*Nvaccine*Nstrata, Nstrata):
        np.fill_diagonal(Amatrix[i+1:i+Nstrata, i:i+Nstrata-1],  lamb_vector[i/Nstrata]) #lower diagonal
        np.fill_diagonal(Amatrix[i:i+Nstrata, i:i+Nstrata], np.diag(Amatrix[i:i+Nstrata, i:i+Nstrata]) -lamb_vector[i/Nstrata] -Vwanning_t[i/Nstrata-Nage]-np.asarray([float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]))
        np.fill_diagonal(Amatrix[i:i+Nstrata-1, i+1:i+Nstrata], np.asarray([float(j+1)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)])) #upper diagonal 
        Amatrix[i:i+Nstrata,i-Nage*Nstrata:i-Nage*Nstrata+Nstrata] =VRate[i/Nstrata-Nage]*np.identity(Nstrata) 

    """ Use updated Amatrix """
    #man  and snail equations:
    output[xrange(flattened_size)] = np.dot(Amatrix, hstrata) + Sources
    output[flattened_size] = Lambda_vector.sum()*(1.-ystate.sum())-sigma*ystate[0]
    output[flattened_size+1] =  sigma*ystate[0] -epsi*ystate[1]

    return output


#@profile
#def main(equi_h=[], AgeFrac=[], Events=[], AayF=1., BeeF=1., gammaF=1., Aay_, Bee_, rho0_, sigma=1., epsi=2.0, deldblu=2, SimTime=2., Nstrata=100, zstate_0=0.01, FracVaccAtCH=0.0, Vwanning= 1./3., VEff_fecund=0.0, VEff_sus=0.0, VEff_wmort=0.0, TE_wmort=0.0):
def RunSim(equi_h, AgeFrac, Events, Aay_, Bee_, rho0_, AayF=1., BeeF=1., gammaF=1., sigma=1., Pop0=2000., epsi=2.0, deldblu=2, SimTime=2., Nstrata=100, zstate_0=0.01, FracVaccAtCH=0.0, Vwanning= 1./3., VEff_fecund=0.0, VEff_sus=0.0, VEff_wmort=0.0, TE_wmort=0.0):

    Agenames=np.array(['CH','SAC','YAdults','Adults'])
    Agebins=np.array([5,10,15,70])
    AgeMort=np.array([0.074, 0.007, 0.025, 0.06])
    gamma_=np.array([1./5.7]*4)
    kay0_=np.array([7000.]*4)
    ystate_0 = 0.9

#    #Events{GVaccinated=np.empty(0), FVaccinated=0.9, VRate}
#    Events = {}
#    FVaccinated=0.75
#    Vscaleup=0.03
#    Vwanning = 1./(0.2)
#    # exponentially distributed rate of vaccination
#    VRate = np.empty(Nage)
#    VRate.fill(-np.log(1. - FVaccinated)/Vscaleup)
#    x=np.linspace(0,100,1200)
#    y=np.append(np.linspace(VRate[0],0,2), np.zeros(1200-2))
#    from scipy.interpolate import interp1d
#    Func_VRate=interp1d(x,y)
#    #efficacy=86.3%
#    VEff_wmort =(-np.log(1.-0.863)/(28./365.))/gamma_[0]-1. #from (1+VEm)*gamma=surviving fraction of worms over a given period 
#    for i in np.arange(1,5,0.5): 
#        Events[i] = {'GVaccinated':np.array(range(Nage)),'VRate':Func_VRate}



    #GVaccinated = np.array(GVaccinated)

    Aay_ = Aay_*AayF
    Bee_ = Bee_*BeeF
    print Aay_,"\n", Bee_
    gamma_= gamma_*gammaF
    print Nstrata
    Nage = len(Agebins)
    Nvaccine = 2
    maturation = 1./Agebins
    turnover = maturation + AgeMort
    Dparams={}
    Dparams['Pop0']=Pop0
    Dparams['Nstrata']=Nstrata
    Dparams['Nage']=Nage
    Dparams['Mortrate']=AgeMort
    Dparams['Matrate'] = maturation

#    #Obtain age fractions for stable demography
#    if ~os.path.isfile("EqmDemography.npz"):
#        AgeFrac, equi_h = ObtainDemographic(params=Dparams)
#    else:
#        npzfile = np.load("EqmDemography.npz")
#        AgeFrac = npzfile['AgeFrac']
#        equi_h = npzfile['equi_h']


    #create shadow vaccine groups:
    for i in xrange(0,Nage):
        Agenames = np.append(Agenames,"V"+Agenames[i])

    # initial values for the combined system and populate SWBs
    Sources = np.zeros(Nage*Nstrata*Nvaccine)
    SWBs={}
    for i, elm in enumerate(Agenames):
        if i<Nage:
            SWBs[elm]=SWBPOP(Mortrate=AgeMort[i], Nstrata=Nstrata, deldblu=deldblu, Matrate=maturation[i], Fpop=AgeFrac[i], mu=turnover[i], Bee=Bee_[i], Aay=Aay_[i], gamma=gamma_[i], rho0=rho0_[i], kay0=kay0_[i])
            SWBs[elm].State = np.hstack((AgeFrac[i]*equi_h.sum(),np.zeros(SWBs[elm].p['Nstrata']-1)))
            SWBs[elm].nn = np.arange(SWBs[elm].p['Nstrata'],dtype=float)
        else:
            SWBs[elm]=SWBPOP(Mortrate=AgeMort[i-Nage], Nstrata=Nstrata, deldblu=deldblu, Matrate=maturation[i-Nage], mu=turnover[i-Nage], Bee=Bee_[i-Nage], Aay=Aay_[i-Nage], gamma=gamma_[i-Nage], rho0=rho0_[i-Nage], kay0=kay0_[i-Nage], VEff_fecund=VEff_fecund,VEff_sus=VEff_sus,VEff_wmort=VEff_wmort)
            SWBs[elm].State = np.zeros(SWBs[elm].p['Nstrata'])
            SWBs[elm].nn = np.arange(SWBs[elm].p['Nstrata'],dtype=float)

    """----------- Define fixed part of the matrix of Coefficients --------------------------------------"""
    # each of the _vector quantities are of size Nage*Nvaccine
    #gamma_vector = np.append(gamma_, (1.+VEff_wmort)*gamma_) #vaccinated could have heightened worm mortality
    AgeMort_vector = np.append(AgeMort, AgeMort) 
    Maturation_vector = np.append(maturation, maturation)
    Adim=Nage*Nstrata*Nvaccine
    Amatrix_fixed = np.zeros((Adim, Adim))  #fixed part of matrix of coefficients
    """ Unvaccinated """
    #Unvaccinated CH
    i = 0
    L_diag = np.zeros(Nstrata)
    #Main_diag = [-(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
    Main_diag = -(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata])*np.ones(Nstrata)
    #Main_diag = np.asarray(Main_diag) 
    #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
    U_diag = np.zeros(Nstrata) 
    #U_diag = np.asarray(U_diag)
    data=np.vstack((L_diag, Main_diag, U_diag))
    diags = np.array([-1, 0, 1])
    Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()


    # unvaccinated in between
    for i in xrange(Nstrata, (Nage-1)*Nstrata, Nstrata):
        #Main_diag = [-(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
        Main_diag = -(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata])*np.ones(Nstrata) 
        #Main_diag = np.asarray(Main_diag) # -lamb_vector[i/Nstrata] -VRate_t[i/Nstrata]   #vaccine updates
        #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
        U_diag = np.zeros(Nstrata) 
        #U_diag = np.asarray(U_diag)
        data=np.vstack((L_diag, Main_diag, U_diag))
        diags = np.array([-1, 0, 1])
        Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()
        Amatrix_fixed[i:i+Nstrata, i-Nstrata:i] = Maturation_vector[i/Nstrata-1]*np.identity(Nstrata)


    # unvaccinated old
    i=(Nage-1)*Nstrata
    #Main_diag = [-(AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
    Main_diag = -(AgeMort_vector[i/Nstrata])*np.ones(Nstrata)
    #Main_diag = np.asarray(Main_diag) #-lamb_vector[i/Nstrata] -VRate_t[i/Nstrata]   #vaccine updates
    #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
    U_diag = np.zeros(Nstrata) 
    #U_diag = np.asarray(U_diag)
    data=np.vstack((L_diag, Main_diag, U_diag))
    diags = np.array([-1, 0, 1])
    Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()
    Amatrix_fixed[i:i+Nstrata, i-Nstrata:i] = Maturation_vector[i/Nstrata-1]*np.identity(Nstrata)
    


    """ Vaccinated """
    #vaccinated CH
    i = Nage*Nstrata

    #Main_diag = [-(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
    Main_diag = -(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata])*np.ones(Nstrata) 
    #Main_diag = np.asarray(Main_diag) #-lamb_vector[i/Nstrata] -Vwanning_t[i/Nstrata-Nage]  #vaccine updates
    #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
    U_diag = np.zeros(Nstrata) 
    #U_diag = np.asarray(U_diag)
    data=np.vstack((L_diag, Main_diag, U_diag))
    diags = np.array([-1, 0, 1])
    Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()


    # vaccinated in between
    for i in xrange(Nstrata+Nage*Nstrata, (Nage*Nvaccine-1)*Nstrata, Nstrata):
        #Main_diag = [-(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
        Main_diag = -(Maturation_vector[i/Nstrata]+AgeMort_vector[i/Nstrata])*np.ones(Nstrata)
        #Main_diag = np.asarray(Main_diag) #-lamb_vector[i/Nstrata] -Vwanning_t[i/Nstrata-Nage]  #vaccine updates 
        #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
        U_diag = np.zeros(Nstrata)
        #U_diag = np.asarray(U_diag)
        data=np.vstack((L_diag, Main_diag, U_diag))
        diags = np.array([-1, 0, 1])
        Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()
        Amatrix_fixed[i:i+Nstrata, i-Nstrata:i] = Maturation_vector[i/Nstrata-1]*np.identity(Nstrata)


    # vaccinated old
    i=(Nage*Nvaccine-1)*Nstrata
    #Main_diag = [-(AgeMort_vector[i/Nstrata]+float(j)*gamma_vector[i/Nstrata]) for j in xrange(Nstrata)]
    Main_diag = -(AgeMort_vector[i/Nstrata])*np.ones(Nstrata) 
    #Main_diag = np.asarray(Main_diag) #-lamb_vector[i/Nstrata] -Vwanning_t[i/Nstrata-Nage]    #vaccine updates
    #U_diag = [float(j)*gamma_vector[i/Nstrata] for j in xrange(Nstrata)]
    U_diag = np.zeros(Nstrata) 
    #U_diag = np.asarray(U_diag)
    data=np.vstack((L_diag, Main_diag, U_diag))
    diags = np.array([-1, 0, 1])
    Amatrix_fixed[i:i+Nstrata, i:i+Nstrata] = spdiags(data, diags, Nstrata, Nstrata).toarray()
    Amatrix_fixed[i:i+Nstrata, i-Nstrata:i] = Maturation_vector[i/Nstrata-1]*np.identity(Nstrata)

    """-------------------END of Amatrix_fixed definition -------------------------------"""


    AgeMort_v = np.array([elm*np.ones(Nstrata) for i,elm in enumerate(AgeMort_vector)]).flatten()
    #faster in place substitution at Sources[0]:
    Sources.itemset(0, np.sum(AgeMort_v*np.append(equi_h, np.zeros(equi_h.shape)))) #source to uninfected unvaccinated CH


    #calculate phi for one SWB, no difference across age/vaccinated classes:
    phi_k = SWBs[Agenames[0]].phi()
    hstrata = np.append(equi_h,np.zeros(equi_h.shape))
    State0=np.append(hstrata,np.array([ystate_0, zstate_0])) #using previously defined demographic state and initial snail prevalence
    #stabilizing baseline
    burntime = np.linspace(0,10,20)
    State_burnBL = integrate.odeint(dState, State0, burntime, args=(Sources, Amatrix_fixed, Aay_, Bee_, gamma_, rho0_, kay0_, phi_k, Nstrata, deldblu, Nage, Nvaccine, sigma, epsi, Vwanning, None, None, VEff_fecund, VEff_sus, VEff_wmort, TE_wmort))
    State_burnBL = State_burnBL[-1,:]
    #sort events according to time
    VaccineTimes = np.sort(Events.keys())
    print "Vaccination times=", VaccineTimes
    #up to first time nothing happens
    time1 = np.linspace(0,VaccineTimes[0],52)
    State1 = integrate.odeint(dState, State_burnBL, time1, args=(Sources, Amatrix_fixed, Aay_, Bee_, gamma_, rho0_, kay0_, phi_k, Nstrata, deldblu, Nage, Nvaccine, sigma, epsi, Vwanning, None, None, VEff_fecund, VEff_sus, VEff_wmort, TE_wmort))

    #introduce Vaccine
    # vaccination at birth if FracVaccAtCH is not zero
    Sources.itemset(0, (1.-FracVaccAtCH)*np.sum(AgeMort_v*np.append(equi_h, np.zeros(equi_h.shape)))) #source to uninfected unvaccinated CH
    Sources.itemset(Nage*Nstrata, FracVaccAtCH*np.sum(AgeMort_v*np.append(equi_h, np.zeros(equi_h.shape)))) #source to uninfected vaccinated CH

    State = State1.copy()
    State1 = State1[-1,:]
    time = time1.copy()
    for tind in xrange(VaccineTimes.shape[0]-1):
        time_dummy = np.linspace(VaccineTimes[tind],VaccineTimes[tind+1], 100.0)
        time = np.append(time, time_dummy)
        State_dummy = integrate.odeint(dState, State1, time_dummy, args=(Sources, Amatrix_fixed, Aay_, Bee_, gamma_, rho0_, kay0_, phi_k, Nstrata, deldblu, Nage, Nvaccine, sigma, epsi, Vwanning, VaccineTimes[tind], Events[VaccineTimes[tind]], VEff_fecund, VEff_sus, VEff_wmort, TE_wmort))
        State = np.vstack((State, State_dummy))
        State1 = State_dummy[-1,:]

    timef = np.linspace(VaccineTimes[-1], VaccineTimes[-1]+SimTime, SimTime*52.)
    time = np.append(time, timef)
    Statef = integrate.odeint(dState, State1, timef, args=(Sources, Amatrix_fixed, Aay_, Bee_, gamma_, rho0_, kay0_, phi_k, Nstrata, deldblu, Nage, Nvaccine, sigma, epsi, Vwanning, VaccineTimes[-1], Events[VaccineTimes[-1]], VEff_fecund, VEff_sus, VEff_wmort, TE_wmort))
    
    State = np.vstack((State, Statef))
    
    ystate = State[:,-2]
    zstate = State[:,-1]
    hstate = State[:,:-2]
    for i, elm in enumerate(Agenames):       #going through the age blocks in hstrata (every Nstrata)
        SWBs[elm].getStates(times=time,States=hstate[:,i*Nstrata:(i+1)*Nstrata])

    return time, SWBs, hstate, ystate, zstate

##allow this code to be run as a script
#if __name__ == "__main__":
#    print "Usage %run VaccineModel.py"
#    time, SWBs, hstate, ystate, zstate =  main(equi_h=[], AgeFrac=[], Events=[], AayF=1., BeeF=1., gammaF=1., Aay_, Bee_, rho0_, sigma=1., epsi=2.0, deldblu=2, SimTime=2., Nstrata=100, zstate_0=0.01, FracVaccAtCH=0.0, Vwanning= 1./3., VEff_fecund=0.0, VEff_sus=0.0, VEff_wmort=(-np.log(1.-0.863)/(28./365.))*5.7-1., TE_wmort=0.0):
