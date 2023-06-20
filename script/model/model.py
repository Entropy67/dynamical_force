## -----------------------------------
## -----------------------------------

"""
Stochastic simulation for binding and unbinding reactions
    File name: model.py
    Author: Hongda Jiang
    Date created: 10/14/2019
    Date last modified: 10/14/2019
    Python Version: 3.6
    Requirst package: Numpy, Matplotlib, Random

    Log:
    10142019: created the file
"""

__author__ = "Hongda Jiang"
__copyright__ = "Copyright 2018, UCLA PnA"
__license__ = "MIT"
__email__ = "hongda@physics.ucla.edu"
__status__ = "Building"

## ----------------------------------
## ----------------------------------




import numpy as np
from math import log
import copy
import heapq
import matplotlib.pyplot as plt
import timeit
from . import utilities as utl


prm_dict = {
    ### config
    #"init_config": [10, 0, 10, 0], ### initial bond config, [m1, n1, m2, n2]

    ### force parameter
    "scheme":"hillT", ## ramp, const, step, coop, hill

    "method": "gillespie",
    "loadSharing": True,
    "beta":1.0,
    "mc":100,
    "f0": 600, ## in pN
    "tc":120000,

    ##### bonds parameters
    "tau_a": [300*1000, 300*1000], ## in second
    "tau_b": [600*1000, 600*1000], ## in second
    "xb1": 1.5,
    "xb2": 2.0,
    "ton": 20000,
    "l1": 0,
    "l0":100,

    ### other parametersL
    "tm":30,

    "tmin":0,

    "enzymic_deteach":True
    }



###
kT = 4.012





class SSA:
    """container for stochastic simulation algorithm"""

    def __init__(self, model, seed=None, method="next_reaction"):
        self.model = model
        if seed is not None:
            np.random.seed(seed)

        self.method = method ## method = "gillespie", "next_reaction", "modified_next_reaction"
        ## see David F. Anderson, The Journal of Chemical Physics 127, 214107 (2007)
        self.build()
        return

    def build(self):
        try:
            self.num_rect = self.model._get_reaction_num()
        except:
            self.num_rect = len(self.model.get_propensity())

        if self.method == "next_reaction":
            self.tau_id = np.empty((self.num_rect), dtype=list)
            self.tau = None ## heap store tau_k

        if self.method == "modified_next_reaction":
            self.P = np.asarray([ log(1.0/np.random.random()) for _ in range(self.num_rect)])
            self.T = np.zeros((self.num_rect))
            self.deltaT = None

        #self.model.setup()
        return


    def run(self, output=True, check=None):
        """indefinite generator of 1st-reaction trajectories"""
        self.build()
        if self.method=="gillespie":
            reaction = self.gillespie
        elif self.method =="next_reaction":
            reaction = self.next_reaction

        elif self.method=="modified_next_reaction":
            if self.model.time_dependent:
                reaction = self.modified_next_reaction_time_dependent
            else:
                reaction = self.modified_next_reaction

        else:
            return

        step = 0
        while not self.model.exit():
            step += 1
            reaction()
            if step%100==0 and check is not None:
                if not check("(sto.SSA): "):
                    return
        return


    def gillespie(self):

        ### get propensity
        a = self.model.get_propensity()
        if(min(a)<0):
            raise Exception("*** SSA.gillepsie: Error: a<0:", a)

        ### sample the waiting
        a0 = sum(a)
        if(a0==0):
            raise Exception("****SSA.gillespie: Error: a0=0")

        #r1 = np.random.random()
        T = log(1.0/np.random.random())/a0


        ### sample next reaction type
        next_r = 0
        S = np.random.random()*a0 - a[next_r]
        while(S>0):
            next_r += 1

            try:
                S -= a[next_r]
            except:
                raise Exception("*** SSA.gillepsie: error: S<0")

        self.model.react(T, next_r)
        return

    def next_reaction(self):

        if self.tau is None:
            # print(self.tau_id)
            ### init tau list
            a = self.model.get_propensity()
            self.tau = []
            for i in range(self.num_rect):
                if a[i]>0:
                    self.tau_id[i] = [self.__choose_tau(a[i]), i, a[i]]
                    heapq.heappush(self.tau, self.tau_id[i])
                else:
                    self.tau_id[i] = [-1, i, 0]
        ### get the next reaction
        T, next_r, _ = heapq.heappop(self.tau)

        ### update the tau list
        self.tau_id[next_r] = [-1, next_r]

        ### update the system
        self.model.react(T, next_r, 'b')

        ### recalculate the propensity function
        a_new = self.model.get_propensity()


        for i in range(self.num_rect):
            ### for each k != next_r, set tau_k = (ak/anewk)(tauk - T)+T
            if i!= next_r:
                if self.tau_id[i][0] < 0: ## if i-th reaction is not in the tau heap
                    if a_new[i] >0:
                        self.tau_id[i] = [self.__choose_tau(a_new[i])+T, i, a_new[i]]
                        heapq.heappush(self.tau, self.tau_id[i])
                else:

                    k = self.tau.index(self.tau_id[i])
                    if self.tau[k][1] != i:
                        raise Exception("SSA:next_reaction:error"("index error, tau[k][1]={0:d}, tauList[i][1]={1:d}".format(k, self.tau_id[i][1])))

                    taui_old = self.tau_id[i][0]

                    if a_new[i] >0:
                        ### consider special cae: a_old = 0
                        a_old = self.tau_id[i][2]
                        if a_old>0:
                            taui_new = a_old*(taui_old - T)/a_new[i] + T
                        else:
                            taui_new = self.__choose_tau(a_new[i])+T

                        self.tau_id[i][0] = taui_new
                        self.tau_id[i][2] = a_new[i]

                        if taui_new > taui_old:
                            heapq._siftup(self.tau, k)
                        else:
                            heapq._siftdown(self.tau, 0, k)
                    else:
                        ### if a_new = 0, we would like to remove the node from heap
                        self.tau[k] = self.tau[-1]
                        self.tau.pop()

                        self.tau_id[i] = [-1, i, 0]

                        if k<len(self.tau):
                            heapq._siftdown(self.tau, 0, k)
                            heapq._siftup(self.tau, k)
            else: ### for the current reaction, update tau = tau(a_new)+T
                if a_new[i]>0:
                    self.tau_id[i] =[self.__choose_tau(a_new[i])+T, i, a_new[i]]
                    heapq.heappush(self.tau, self.tau_id[i])
        #print(self.tau)
        return

    def __choose_tau(self, ai):
        if ai==0:
            print("*****SSA.__choose_tau: WARRING:  divieded by 0 when finding T****")
            raise Exception()
        # choose the increment T in time as (1/a0)*ln(1/r1)
        tau = log(1.0/np.random.random())/ai
        if tau == 0:
            print("*****SSA.__choose_tau:WARRING:  tau is 0!!!****")
        return tau

    def modified_next_reaction(self):
        a = self.model.get_propensity() ###
        deltaT = []
        dt, mu = np.inf, -1 ### time interval and next reaction

        dt_min = np.inf
        for k in range(self.num_rect):
            dtk = (self.P[k] - self.T[k])/a[k] if a[k]>0 else np.inf
            if dtk<dt_min:
                dt_min, mu = dtk, k
            deltaT.append(dtk)

        self.model.react(dt_min, mu, 'a') ## 'a' means additive, the dt is waiting time interval
        for k in range(self.num_rect):
            self.T[k] += a[k]*dt_min
        self.P[mu] += log(1.0/np.random.random())
        return

    def modified_next_reaction_time_dependent(self, dt=0.01):
        a = self.model.get_propensity() ### a[k]>0 time-indepedent reaction, a[k]=0 no reaction, a[k]<0 time-dependent reaction
        deltaT = [np.inf for _ in range(self.num_rect)]
        dt, mu = np.inf, -1 ### time interval and next reaction

        dt_min = np.inf
        k_list = [] ### time-dependent reaction list
        for k in range(self.num_rect):
            if a[k] == 0:
                dtk = np.inf
            elif a[k]>0:
                dtk = (self.P[k] - self.T[k])/a[k]
            else:
                k_list.append(k)
                continue
            if dtk<dt_min:
                dt_min, mu = dtk, k
            deltaT[k] = dtk

        for k in k_list:
            dtk = self.model.integrate_propensity(k, dt_min, self.P[k] - self.T[k]) ### solve int_t^(t+dtk) a[k](s)ds = P[k]-T[k]
            if dtk<dt_min:
                dt_min, mu = dtk, k
            deltaT[k] = dtk


        for k in range(self.num_rect):
            if a[k]>=0:
                self.T[k] += a[k]*dt_min
            else: ### time-dependent reaction rate
                self.T[k]+= self.model.integrate_propensity(k, dt_min)
        self.P[mu] += log(1.0/np.random.random())

        self.model.react(dt_min, mu, 'a') ## 'a' means additive, the dt is waiting time interval
        return


class Base_Model:

    def __init__(self):
        self.rates = [0.001, 0.002]
        self.init_cond = [100, 100]
        #self.setup()
        pass

    def setup(self):
        ### init condition
        self.current_species = self.init_cond.copy()
        self.num_rect = len(self.rates)
        self.num_spec = len(self.init_cond)
        self.t = 0
        self.history = {"t":[0]}
        for s in range(self.num_spec):
            self.history["m"+str(s+1)] = [self.current_species[s]]
        return

    def get_propensity(self):
        a0 = self.current_species[0]*self.rates[0]
        a1 = self.current_species[1]*self.rates[1]
        a2 = self.rates[2]
        return [a0, a1, a2]

    def exit(self):
        return self.t>0 and self.current_species[0]<=0

    def react(self, T, next_r, mod='a'):
        self.t = self.t + T if mod=='a' else T
        if next_r == 0:
            self.current_species[0] -= 1
            #self.current_species[2] += 1
        elif next_r==1:
            self.current_species[1] -= 1
            #self.current_species[2] -= 1
        else:
            self.current_species[2] += 1

        self._record()
        return

    def _get_reaction_num(self):
        return self.num_rect

    def _record(self):
        self.history["t"].append(self.t)
        for s in range(self.num_spec):
            self.history["m"+str(s+1)].append(self.current_species[s])
        return

    def _get_history(self, qty=None):
        if qty is None:
            return self.history
        else:
            return np.asarray(self.history[qty])

    def _plot_traj(self, qtys=None, ax=None, **keyargs):
        if qtys is None:
            print("please specify species indexes, i.e. qtys=[m1, m2, m3, m4]")
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2), dpi=150)

        for idx in qtys:
            ax.plot(np.asarray(self.history["t"])/60000, self.history[idx], **keyargs)
        return ax

class Model2(Base_Model):

    def __init__(self):
        super().__init__()
        self.time_dependent = True
        self.dt = 0.1 ### time step for integral
        self.lmd = 0.001
        pass


    def get_propensity(self, t=-1):
        a0 = self.current_species[0]*self.rates[0]
        a1 = -1 if t<0 else self.current_species[1]*self.rates[1]*np.exp(-self.lmd*(self.t))
        return [a0, a1]


    def integrate_propensity(self, k , dt_min, dp=None):
        '''if dp is None:
               return int_t^(t+dt_min)a[k](s)ds
           if dp is not None:
               numerically solve the condition: int_t^{t+dT} a[k](s)ds = dp'''
        s, dT = 0, 0
        while (dp is None or s<dp) and dT <= dt_min:
            s += self.current_species[1]*self.rates[1]*np.exp(-self.lmd*(self.t+dT))*self.dt
            dT += self.dt
        if dp is None:
            return s
        else:
            return dT if dT<dt_min else 2*dt_min

    def exit(self):
        return self.t>0 and self.current_species[0]<=0

    def react(self, T, next_r, mod='a'):
        self.t = self.t + T if mod=='a' else T
        if next_r == 0:
            self.current_species[0] -= 1
        else:
            self.current_species[1] -= 1
        self.record()
        return


    def record(self):
        self.history["t"].append(self.t)
        for s in range(self.num_spec):
            self.history["m"+str(s+1)].append(self.current_species[s])
        return

    def get_history(self, qty):
        return self._get_history(qty)


class Force_prm:
    def __init__(self, sto, scheme="hillT"):
        self.sto = sto
        self.scheme=scheme
        ### const, clst dep.
        self.f0 = 0

        ### ramping
        self.r = 0

        ### step, time dep.
        self.tc = 1e5

        ### step, clst dep.
        self.mc = 100

        ### coop
        self.beta = 1.0

        self.f = 0

        self.update_method = 0



    def init(self):
        self.f = 0
        pass

    def loadprm(self, prmdict):

        self.scheme=prmdict["scheme"]
        self.update_method = prmdict["force_update"]

        if self.scheme not in ["hillT", "powT", "step"] and self.update_method !=0:
            raise Exception("force_prm: update method wrong!!! please set it to zero ")
        self.f0 = prmdict["f0"]
        self.mc = prmdict["mc"]
        self.tc = prmdict["tc"]
        self.beta = prmdict["beta"]
        self.kf = prmdict["kf"]
        self.df = prmdict["df"]
        pass

    def is_time_dependent(self):
        return self.scheme in ["hillT", "powT"] and self.update_method==0

    def get_f(self, t,   m_max=1):

        ### f is in pN
        if self.scheme=="const":
            ## clst dep. step force
            if m_max >= self.mc:
                self.f = self.f0
            else:
                self.f = 0

        elif self.scheme=="step":
            ## time dep. step force
            if t >= self.tc:
                self.f = self.f0
            else:
                self.f = 0

        elif self.scheme=="hillT":
            if t<=0:
                self.f = 0
            else:
                if self.update_method == 2:
                    self.f += self.df
                else:
                    self.f = self.f0 * (1/(1+(self.tc/t)**self.beta))

        elif self.scheme=="hillN":
            if t>0:
                self.f = self.f0*( m_max**self.beta / ( m_max**self.beta + (self.mc)**self.beta))
            else:
                self.f = 0

        elif self.scheme=="powT":
            if t<=0:
                self.f = 0
            else:
                if self.update_method == 2:
                    self.f += self.df
                else:
                    self.f = self.f0*(t/self.tc)**self.beta

        elif self.scheme=="powN":
            if  t>0:
                #self.f  =min(self.f0*(m_max/self.mc)**self.beta, self.f0)
                self.f  =self.f0*(m_max/self.mc)**self.beta
            else:
                self.f = 0
        else:
            raise Exception("!!! no such force schemes"+self.scheme)
            self.f = 0
        return self.f

    def get_kf(self, t=0):
        """
        get force chaging rate
        """
        if self.scheme=="hillT":
            if self.update_method == 1:
                return self.kf
            elif self.update_method == 2:
                if t>0:
                    t2 = self.tc*((1.0/(self.df/self.f0 + 1.0/(1.0+(self.tc/t)**self.beta)) - 1)**(-1.0/self.beta))
                    if t2==t:
                        raise Exception("force not updated")
                    return 1.0/(t2-t)
                else:
                    t2 = self.tc/((self.f0/self.df-1)**(1.0/self.beta))
                    return 1.0/t2

        elif self.scheme=="powT":
            return self.kf

        elif self.scheme=="step":
            if t < self.tc/2:
                return 0
            else:
                return self.kf
        else:
            return 0





class Stoch(Model2):

    def __init__(self, prm):
        self.prm = copy.deepcopy(prm)
        self.qty_name = ["tr",
                         "bds",
                         "fr",
                         "m1Max", "m2Max",
                         "mm", "mm1", "mm2",
                         "mc", "nc", "fc",
                         "nr", ### number of reactions
                         "fa",  ## mean rupture force per bond for Ag1
                         "fb", ## mean rupture force per bond for Ag2
                         "m1Tot", ### all antigens that engaged with BCR
                         "m2Tot" ## total number of Ag2 that engaged with BCR
                         ]

        self.dataset = {}
        super().__init__()
        self.rates = self.init_rate()
        self.init_cond = self.init_spec()  ## spec list [m1, n1, m2, n2, f]
        self.storeTraj = True
        self.storeForce = True
        self.finished = True
        self.setup()
        return

    def init_rate(self):
        '''
        initilize reaction rate

        rates[0]: APC-Ag1-BCR ==> APC + Ag-BCR, k1off
        rates[1]: APC + Ag1-BCR ==> APC-Ag-BCR
        rates[2]: APC-Ag1-BCR ==> APC-Ag + BCR
        rates[3]: APC-Ag1 + BCR ==> APC-Ag-BCR
        rates[4]: APC-Ag2-BCR ==> APC + Ag2-BCR
        rates[5]: APC + Ag2-BCR ==> APC-Ag2-BCR
        rates[6]:  APC-Ag2-BCR ==> APC-Ag2 + BCR
        rates[7]: APC-Ag2 + BCR ==> APC-Ag2-BCR
        rates[8]: force update
        '''
        rates = np.zeros(9)
        rates[0] = 1.0/self.prm["tau_a"][0]
        rates[1] = 1.0/self.prm["ton"]

        rates[2] = 1.0/self.prm["tau_b"][0]
        rates[3] = 1.0/self.prm["ton"]

        rates[4] = 1.0/self.prm["tau_a"][1]
        rates[5] = 1.0/self.prm["ton"]

        rates[6] = 1.0/self.prm["tau_b"][1]
        rates[7] = 1.0/self.prm["ton"]

        rates[8] = 0
        return rates

    def init_spec(self, output=False):
        """
        spec[0]: m1
        spec[1]: n1
        spec[2]: m2
        spec[3]: n2
        spec[4]: f
        """
        if "initCond" in self.prm:
            if self.prm["initCond"] == "all":
                return np.asarray([self.prm["l0"], 0, self.prm["l1"], 0, 0.0])
            elif self.prm["initCond"] == "one":
                return np.asarray([min(1, self.prm["l0"]), 0, min(1, self.prm["l1"]), 0, 0.0])

            elif self.prm["initCond"] == "equilibrium":
                m1, n1 = self._sample_steady_state(N=self.prm["l0"],
                                                   kon=1.0/self.prm["ton"],
                                                   ka=1.0/self.prm["tau_a"][0],
                                                   kb = 1.0/self.prm["tau_b"][0], output=output)
                m2, n2 = self._sample_steady_state(N=self.prm["l1"],
                                                  kon=1.0/self.prm["ton"],
                                                   ka=1.0/self.prm["tau_a"][1],
                                                   kb = 1.0/self.prm["tau_b"][1], output=output
                                                  )
                return np.asarray([m1, n1, m2, n2, 0.0])
        return np.zeros(5)


    def setup(self):
        for qty in self.qty_name:
            self.dataset[qty] = []

        super().setup()

        self.force = Force_prm(self)
        self.force.loadprm(self.prm)
        self.time_dependent = self.force.is_time_dependent()

        self.loadSharing = self.prm["loadSharing"]
        if "enzymic_detach" in self.prm:
            self.enzymic_detach = self.prm["enzymic_detach"]
        else:
            self.enzymic_detach = False
        if self.prm["tm"]<1000: ### if tm in the unit of min
            self.tm = self.prm["tm"]*60000
        else: ## if tm in the unit of ms
            self.tm = self.prm["tm"]
        self.tmin = self.prm["tmin"] ## minimal time of contact, 1000ms

        self.l0 = self.prm["l0"]
        self.l1 = self.prm["l1"]

        ## spec list [m1, n1, m2, n2]
        ### setting up the stochastic system
        self.rec_name = []
        self.stoch = np.zeros((self.num_rect, self.num_spec))

        for i in range(2):
            #### APC-Agi-BCR ==> APC + Agi-BCR
            self.rec_name.append("APC-Ag"+str(i)+"_off")
            self.stoch[0+i*4, 0+i*2] = -1
            self.stoch[0+i*4, 1+i*2] = 1

            ### APC + Agi-BCR ==> APC-Agi-BCr
            self.rec_name.append("APC-Ag"+str(i)+"_on")
            self.stoch[1+i*4, 0+i*2] = 1
            self.stoch[1+i*4, 1+i*2] = -1

            ### APC-Agi-BCR ==> APC-Agi + BCR
            self.rec_name.append("BCR-Ag"+str(i)+"_off")
            self.stoch[2+i*4, 0+i*2] = -1
            self.stoch[2+i*4, 1+i*2] = 0

            ## APC-Agi + BCR ==> APC-Agi-BCR
            self.rec_name.append("BCR-Ag"+str(i)+"_on")
            self.stoch[3+i*4, 0+i*2] = 1
            self.stoch[3+i*4, 1+i*2] = 0

        ## force increasement
        self.rec_name.append("force updates")
        self.stoch[-1, 0] = 0

        self.ssa = SSA(self, method=self.prm["sim_method"])
        self.ssa.build()
        self.reset()
        return

    def reset(self):
        self.t = 0
        self.t_start = None ## time of the first ligand binding

        self.f = self.force ## pN
        self.step = 0
        self.force.init()

        self.updateRate()
        self.rates = self.init_rate()
        self.init_cond = self.init_spec()  ## spec list [m1, n1, m2, n2, f]

        super().setup() ## init
        self.history["a"] = [] ### reaction types
        self.history["f"] = [self.f] ### force
        if len(self.history["t"])>1:
            raise Exception("wrong history reset", self.history)
        #### maximum bound complex, variable
        self.m_max =0 ### max cluster size
        self.m1_max = 0 ### max Ag1 cluster size
        self.m2_max = 0 ### max Ag2 cluster size

        self.fn_record = -np.ones(2*(self.l0+self.l1)) ## store rupture force per bond for ag22
        self.fa_record = -np.ones(2*self.l0) ### store rupture force per bond for ag1
        self.fb_record = -2*np.ones(2*self.l1) ### store rupture force per bond for ag2

        if self.init_cond[2]>0:
            m0 = int(self.init_cond[2])
            #idx = np.random.randint(0, self.l1, m0)
            self.fb_record[:m0] = -999## closed
            self.fb_record[self.l1:self.l1+m0] = -999
        if self.init_cond[3]>0:
            m0 = int(self.init_cond[2])
            n0 = int(self.init_cond[3])
            self.fb_record[m0:m0+n0] = 0
            self.fb_record[m0+n0:self.l1] = -1

            self.fb_record[self.l1+m0:self.l1+m0+n0] = -1
            self.fb_record[self.l1+m0+n0:] = 0
        self.finished = False
        return

    def get_propensity(self):
        num_bond = self.current_species[0]+self.current_species[2]
        fn = 0 if num_bond <1 else self.f
        if self.loadSharing and num_bond>0: fn = fn/num_bond
        fn = min(fn, 500) ### cutoff force 500

        a = np.ones(self._get_reaction_num())

        #### APC-Ag1-BCR ==> APC + Ag-BCR
        #### a = k1off * m1
        a[0] = self.rates[0]*self.current_species[0]*np.exp(fn*self.prm["xb1"]/kT)

        ### APC + Ag1-BCR ==> APC-Ag-BCR
        ## a = k1on*n1*(s0 - l10 - l20 + n1 + n2)
        a[1] = self.rates[1]*self.current_species[1]

        ### APC-Ag1-BCR ==> APC-Ag + BCR
        #### a = k2off * m1
        a[2] = self.rates[2]*self.current_species[0]*np.exp(fn*self.prm["xb2"]/kT)

        ## APC-Ag1 + BCR ==> APC-Ag-BCR
        ## a = k2on*(r0-m1-n1-m2-n2)*(l10-m1-n1)
        a[3] = self.rates[3]*(self.l0-self.current_species[0]-self.current_species[1])


        ## APC-Ag2-BCR ==> APC + Ag2-BCR
        ### a = k1off * m2
        a[4] = self.rates[4]*self.current_species[2]*np.exp(fn*self.prm["xb1"]/kT)


        ### APC + Ag2-BCR ==> APC-Ag2-BCR
        ## a = k2on*n2*(s0-l10-l20+n1+n2)
        a[5] = self.rates[5]*self.current_species[3]

        ## APC-Ag2-BCR ==> APC-Ag2 + BCR
        ### a = k2off * m2
        a[6] = self.rates[6]*self.current_species[2]*np.exp(fn*self.prm["xb2"]/kT)

        ### APC-Ag2 + BCR ==> APC-Ag2-BCR
        ## a = k2on*(r0-m1-n1-m2-n2)*(l20-m1-n1)
        a[7] = self.rates[7]*(self.l1-self.current_species[2]-self.current_species[3])

        ### force update
        a[8] = self.force.get_kf(self.t)
        #print("af=", a[8])

        if self.time_dependent and self.prm["force_update"]==0:
            for i in [0, 2, 4, 6]:
                a[i] = -1 if a[i]>0 else 0
            a[8] = 0
        return a

    def integrate_propensity(self, k, dt_min, dp=None):
        if dp is not None and dp<0:
            pass
            #raise Exception("stoch.integrate_propensity: error, dp<0", dp)
        s, dT = 0, 0
        prefactor = self.rates[k]*self.current_species[2*(k//4)] ## 0, 2 -> 0;  4, 6 -> 2
        num_bond = self.current_species[0]+self.current_species[2]

        if k in [0, 4]:
            xb = self.prm["xb1"]
        elif k in [2, 6]:
            xb = self.prm["xb2"]
        else:
            raise Exception("stoch.integrate_propensity: error, integrating time-indepent rate, k=", k)

        num_grid = 400
        if dp is None:
            dt = dt_min/num_grid
        else:
            fn = 0 if num_bond <1 else self.force.get_f(self.t)
            if self.loadSharing and num_bond>0: fn = fn/num_bond
            fn = min(fn, 500)
            dt = dp/(prefactor*np.exp(fn*xb/kT)*num_grid)


        while(dp is None or s<dp) and dT<=dt_min:
            fn = 0 if num_bond <1 else self.force.get_f(self.t+dT)
            if self.loadSharing and num_bond>0: fn = fn/num_bond
            fn = min(fn, 500)
            s += prefactor*np.exp(fn*xb/kT)*dt
            dT += dt

        fn = 0 if num_bond <1 else self.force.get_f(self.t+dT-dt)
        if self.loadSharing and num_bond>0: fn = fn/num_bond
        fn = min(fn, 500)

        if dp is None:
            return s + prefactor*np.exp(fn*xb/kT)*(dt_min - dT+dt)
        else:
            dT = dT-(s-dp)/(prefactor*np.exp(fn*xb/kT))
            return dT if dT<dt_min else 2*dt_min


    def react(self, T, next_r, mod='a'):
        if mod != 'a' and T<self.t:
            raise Exception("stoch.error: T<time.... T=",T, ", time=", self.t, ", next_r=", next_r )

        ### update the time
        self.t = self.t + T if mod=='a' else T


        self.current_species += self.stoch[next_r, :]
        ## update force or species
        if next_r == self.num_rect-1 or self.prm["force_update"]==0: ### reaction for force updating
            self.updateRate()

        self.step +=1

        #self.current_species[-1] = self.f
        if self.t_start is None and (self.current_species[0] + self.current_species[2]>0):
            self.t_start = self.t ### record the time of first binding reaction
        self.record(next_r)
        return

    def record(self, next_r):

        mm =  self.current_species[0]+self.current_species[2] ## current cluster size

        if self.storeTraj:
            self._record() ### store trajectory
            self.history["a"].append(next_r) #### store reaction type
            self.history["f"].append(self.f)
            ##### store force per bond
            if True:
                ll = self.l0 + self.l1
                if next_r == 4:
                    bond_id = np.where(self.fn_record[:ll] == -1)[0]
                    self.fn_record[bond_id[0]] = self.f/(mm+1)
                elif next_r == 6:
                    bond_id = np.where(self.fn_record[ll:] == -1)[0]
                    self.fn_record[ll+bond_id[0]] = self.f/(mm+1)
                elif next_r == 5:
                    bond_id = np.where(self.fn_record[:ll]>-0.5)[0]
                    if len(bond_id)>0:
                        self.fn_record[bond_id[0]] = -1
                elif next_r == 7:
                    bond_id = np.where(self.fn_record[ll:]>-0.5)[0]
                    if len(bond_id)>0:
                        self.fn_record[ll+bond_id[0]] = -1
        if self.storeForce:
            ll = self.l0
            if next_r == 0:
                bond_id = np.where(self.fa_record[:ll] == -1)[0]
                self.fa_record[bond_id[0]] = self.f/(mm+1)
            elif next_r == 2:
                bond_id = np.where(self.fa_record[ll:] == -1)[0]
                self.fa_record[ll+bond_id[0]] = self.f/(mm+1)
            elif next_r == 1:
                bond_id = np.where(self.fa_record[:ll]>-0.5)[0]
                if len(bond_id)>0:
                    self.fa_record[bond_id[0]] = -1
            elif next_r == 3:
                bond_id = np.where(self.fa_record[ll:]>-0.5)[0]
                if len(bond_id)>0:
                    self.fa_record[ll+bond_id[0]] = -1
            ll = self.l1
            if next_r == 4:
                bond_id = np.where(self.fb_record[:ll] <-3)[0]
                self.fb_record[bond_id[0]] = self.f/(mm+1) ### open, visited
                self.fb_record[bond_id[0]+ll] = -1 ### open, visited
            elif next_r == 6:
                bond_id = np.where(self.fb_record[ll:] <-3)[0]
                self.fb_record[ll+bond_id[0]] = self.f/(mm+1) ### open, visited
                self.fb_record[bond_id[0]] = -1 ### open, visited
            elif next_r == 5:
                bond_id = np.random.choice(np.where(self.fb_record[:ll]>-3)[0])
                self.fb_record[bond_id] = -999  ### closed
                self.fb_record[bond_id+ll] = -999

            elif next_r == 7:
                bond_id = np.random.choice(np.where(self.fb_record[ll:]>-3)[0])
                self.fb_record[bond_id] = -999
                self.fb_record[bond_id+ll] = -999

        self.m_max =max(self.m_max, mm)
        self.m1_max = max(self.m1_max, self.current_species[0])
        self.m2_max = max(self.m2_max, self.current_species[2])

    def exit(self):
        if self.t > self.tm:
            self.finished = False
            return True
        elif self.t>self.tmin and (self.current_species[0]+self.current_species[2]==0):
            self.finished = True
            return True
        return False

    def set_prm(self, prm_name, prm_value):
        self.prm[prm_name] = prm_value
        utl.sync_prm(self.prm)
        self.setup()
        return

    def updateRate(self):
        if self.t_start is not None:
            self.m_max =max(self.m_max, self.current_species[0]+self.current_species[2])
            self.f = self.force.get_f(self.t-self.t_start, self.m_max)
        else:
            self.f = self.force.get_f(0, 0)
        #print(self.f)
        return

    def _sample_steady_state(self, N, kon, ka, kb, output=False):
        '''
        sample from the steady state distribution to the force-free master equation
        '''
        gma = kon/(ka + kb)
        eta= ka / (ka+kb)

        if output:
            print("N=", N, ", kon=", kon, ", ka=", ka, ", kb=", kb)
            print("gma=", gma)
            print("eta=", eta)
            print("mean m=", N*kon/(kon+ka+kb))

        m = np.random.binomial(N, gma/(1+gma))
        n = np.random.binomial(N-m, eta)
        return m, n

    def _print2(self, s):
        print(s)
        return

    def append(self, qty, value):
        self.dataset[qty].append(value)
        return

    def get_mean(self, qty):
        return np.mean(self.dataset[qty], axis=0)

    def get_std(self, qty):
        return np.std(self.dataset[qty], axis=0)

    def collect_data(self, tr, bdr):
        self.append("tr", float(tr))
        bds =[int(bd) for bd in bdr[:4]]
        self.append("bds", bds)
        self.append("fr", float(self.force.f))
        self.append("m1Max", int(self.m1_max))
        self.append("m2Max", int(self.m2_max))
        self.append("mm", int(self.m_max))

        self.append("fa", np.mean(self.fa_record[self.fa_record>-0.5]))
        self.append("fb", np.mean(self.fb_record[self.fb_record>-0.5]))

        ma = bds[0]
        mb = bds[2]

        self.append("m1Tot", len(self.fa_record[self.fa_record>-0.5]) + ma)
        self.append("m2Tot", len(self.fb_record[self.fb_record>-0.5]) + mb)
        pass

    def get_eta_moment(self, alpha=1):
        '''get eta moment over the rupture force per bond distribution'''
        ea, eb = self.prm["ea"], self.prm["eb"]
        def eta(f):
            return 1.0/(1+np.exp(ea-eb+f*0.5/4.012))

        ret = []
        for fb in self.fb_record[self.fb_record>-0.5]:
            ret.append(eta(fb)**alpha)
        return np.mean(ret)

    def get_mean_force_per_bond(self):
        return np.mean(self.fa_record[self.fa_record>-0.5]), np.mean(self.fb_record[self.fb_record>-0.5])

    def get_mtot(self):
        ma = self.current_species[0]
        mb = self.current_species[2]
        return len(self.fa_record[self.fa_record>-0.5])+ma, len(self.fb_record[self.fb_record>-0.5])+mb

    def run(self, n=1000, output=True, storeForce=True):

        for i in range(n):
            self.storeTraj = False
            self.storeForce = storeForce
            self.reset()
            if output:
                pass
                #print("init condition=", self.current_species)
                #print("time =", self.t)
                #print("propensity =", self.get_propensity())
            self.ssa.run(output=output)
            #print("finished=", self.finished, end='\n\n')
            if self.finished or self.enzymic_detach:
                self.collect_data(self.t, self.current_species) ## convert time to second
            else:
                if output:
                    print("not finished!")
                return np.nan, np.nan, np.nan
            if output:
                self.printProgress(i, n)

        self.nbar = self.get_mean("bds")
        self.nstd = self.get_std("bds")
        t_mean = self.get_mean("tr")
        f_mean = self.get_mean("fr")

        if output:
            print("count=", n)
            print("nbar={0:.5f}, {4:.5f}, nstd={1:.5f}, {5:.5f}, t={2:.2f}, f={3:.3f}".format(self.nbar[1], self.nstd[1], t_mean, f_mean,self.nbar[3], self.nstd[3]))
        return self.nbar, self.nstd, t_mean

    def run1(self, output=True, storeData=True, check=None):
        """
        output: print or not
        storeData: store data to history or not
        check: external check function, bool check()
        """
        self.storeTraj = storeData
        if self.storeTraj:

            self.reset()
            if output:
                print("init condition=", self.current_species)
            self.ssa.run(output=output)

            if output:
                print("SSA done!")
                print("final spec=", self.current_species)
            self.t_record = np.asarray(self.history["t"])
            self.spec_record = np.asarray([ [self.history["m1"][i],self.history["m2"][i],self.history["m3"][i],self.history["m4"][i]] for i in range(len(self.history["t"]))])
            self.f_record = np.asarray(self.history["f"])
            return
        else:
            self.reset()
            self.ssa.run(output=output, check=check)
            if self.enzymic_detach or self.finished:
                return True, self.t, self.current_species, self.f
            else:
                if output: print("unfinished")
                return False, np.nan, np.nan, np.nan

    def printProgress(self, n, N):
        percent = int(100.0*n/N)
        toPrint = "progress: "
        for i in range(percent//5):
            toPrint += '|'
        toPrint += "{:d}%    ".format(percent)
        print(toPrint, end='\r')
        return



    pass
