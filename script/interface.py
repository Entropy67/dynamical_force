"""
interface connecting scan and model
"""
import numpy as np
import logging
import copy
import numpy.random as rd

import model.model as sim
import data
import controller
from utilities import *

class Agent(controller.Controller):


    def __init__(self, prm, jobs=["spd", "eb_sens", "l0_sens", "antag", "fidelity", "opt_fid"]):
        super().__init__(prm)
        self.jobs = jobs

        self.saved_qty = [
                         "tr", "tstd","fr", "fstd", "tr_all",
                         "nag", "nstd", "nag_antag", "nstd_antag", "nag_all",
                         "m2max", "m2mstd", "m1max", "m1mstd", "mmax", "mmstd", "m2max_all",
                         "tr_most_prob", "tr_median",
                         "ntot", "nstd_tot",
                         "m1tot", "m2tot",
                         "fra", "frb",
                         "m1stdtot", "m2stdtot", "fstdra", "fstdrb",
                         "eta_bar",### averaging all rupture process
                         "eta_var", ### averaging all rupture process
                         "etastd_bar", "etastd_var",

                         "nag1", "nag2", "nstd1", "nstd2",
                         "nag3", "nag4", "nstd3", "nstd4",
                         "eb_sens", "l0_sens", "eb_acc", "l0_acc",
                         "log_eb_sens", "log_l0_sens",

                          "nag0","fr0", "tr0", "m2max0", "nstd0",
                          "fstd0", "tstd0", "m2mstd0",
                          "m1tot0", "m2tot0", "fra0", "frb0",
                          "antagN", "antagT",

                         "nag5", "nag6",
                         "fidelity_n", "fidelity_t", "fidelity_m",

                         "Dkl",

                         "opt_prm",
                         "opt_acc",
                         "opt_prm_0", "opt_prm_1", "opt_prm_2",

                        "nag_seg", "fr_seg", "tr_seg", "m2max_seg", "nag_antag_seg", "m1max_seg", "mmax_seg",
                        "nstd_seg", "fstd_seg", "tstd_seg", "m2mstd_seg", "nstd_antag_seg", "m1mstd_seg", "mmstd_seg"
                     ]


        add_config = {"num_run": 10,
            "cluster_num" : 20,
            "de":0.5,
            "dl":10,
            "mini_surv_chance": 1.0
            }


        if "opt_fid" in jobs:
            add_config.update(
            {
                "opt_prm_names": "f0",
                "opt_bounds": [0, 2000],
                "opt_step_size": 100,
                "opt_n_iterations": 10,
                "opt_temp": 10,
                "opt_prm_init": 400
            }
            )
        self.update_config(add_config)

        self.tag = "Agent"
        self.color = "magenta"

        self.build()
        return


    def build(self):
        super().build()
        self.sto = sim.Stoch(self.prm)

        self.num_run = self.get_config("num_run")
        self.cluster_num = self.get_config("cluster_num")
        self.ns = int(1/self.get_config("mini_surv_chance"))

        self.de = self.get_config("de")
        self.dl = self.get_config("dl")

        self.eb0 = self.prm["eb"]
        self.l0 = self.prm["l0"]
        self.l1 = self.prm["l1"]

        if "l1std" in self.prm:
            self.l1std = self.prm["l1std"]
        else:
            self.l1std = 0

        if "opt_fid" in self.jobs:
            self.opt_prm_names = self.get_config("opt_prm_names")
            self.opt_prm_init = self.get_config("opt_prm_init")
            self.opt_prm_bounds = self.get_config("opt_bounds")
            self.opt_step_size = self.get_config("opt_step_size")
            self.opt_n_iterations = self.get_config("opt_n_iterations")
            self.opt_temp = self.get_config("opt_temp")
            self.last_best = None
        return


    def set_prm(self, prm_name, prm_value):
        self.prm[prm_name] = prm_value
        sync_prm(self.prm)
        self.sto.set_prm(prm_name, prm_value)
        self.build()
        return

    def sample_l0(self):
        """return a sampled l0 according to a Gaussian distribution"""
        rd = np.random.normal(self.l1, self.l1std)
        while rd<=1:
            rd = np.random.normal(self.l1, self.l1std)
        return round(rd)

    def run(self):
        self.print_info()

        ret = True

        if "spd" in self.jobs:
            ### run for speed
            self.print2("speed --------")
            ret = self.run_many()

        if "eb_sens" in self.jobs:
            ###### working on Eb sensitivity
            self.print2("Eb sens: -----")
            n_ag1, n_ag2, n_std1, n_std2, sens, acc, log_sens = self.perturb("eb", self.eb0, self.de, log_prm=False)
            self.dataset.append(
                qtys=["eb_sens", "nag1", "nag2", "nstd1", "nstd2", "eb_acc", "log_eb_sens"],
                values=[sens, n_ag1, n_ag2, n_std1, n_std2, acc, log_sens])
            self.sto.set_prm("eb", self.eb0)

        if "l0_sens" in self.jobs:
            ###### working on L0 sensitivity
            self.print2("looking at the L0 sens: -----")
            n_ag1, n_ag2, n_std1, n_std2, sens, acc, log_sens = self.perturb("l1", self.l1, self.dl, log_prm=True)
            self.dataset.append(
                qtys=["l0_sens", "nag3", "nag4", "nstd3", "nstd4", "l0_acc", "log_l0_sens"],
                values=[sens, n_ag1, n_ag2, n_std1, n_std2, acc, log_sens])
            self.sto.set_prm("l1", self.l1)

        if "antag" in self.jobs:
            self.print2("antag : ------")
            self.get_antag()


        if "fidelity" in self.jobs:
            self.print2("selection fidelity: ---------")
            self.get_fidelity()

        if "random_l0" in self.jobs:
            self.print2("random l0: --------")
            self.get_fidelity(random_l0 = True)

        if "opt_fid" in self.jobs:
            #self.print2("optimizing discrimination fidelity: ")
            self.print2("---------------------- to find the optimal fidelity -----------------")
            self.print2("opt_prm_name: %s" % self.opt_prm_names)
            self.optimize_fidelity()
            #self.find_optimal(output=True)

        if "cross_reactive" in self.jobs:
            self.print2("compare cross-reactive vs specific: ")
            self.run_mixed_ag()
        return ret



    def optimize_fidelity(self):
        output=True

        for n in range(self.num_run):
            self.print2(">> run # %d" % n)
            best, _ = self.find_optimal(output=output)
            output=False
            self.append(["opt_prm_"+str(i) for i in range(len(best[0]))], best[0])
            self.print2(">> optimal parameter: %s" % best)
            self.append(["opt_acc"], [ best[1]])

            if not self.check("(optimize_fidelity):"):
                return

        return

    def run_mixed_ag(self):
        ### run mixed Ag
        print_example = True
        has_nag_data = False

        l0, l1 = self.prm["l0"], self.prm["l1"]
        for i in range(self.num_run):
            if not self.check("(run mixed ag):"):
                return

            self.prm["l0"], self.prm["l1"] = l0, l1

            flag, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, _, _, _, _ = self._run_sim(save_all=True, print_example=print_example)


            if len(ns)==0:
                ns.append(np.nan)
            else:
                has_nag_data = True
            if len(n1s)==0: n1s.append(np.nan)

            if flag:
                self.append(["nag", "fr", "tr", "m2max", "nag_antag", "m1max", "mmax"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="mean")
                self.append(["nstd", "fstd", "tstd", "m2mstd", "nstd_antag", "m1mstd", "mmstd"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="std" )

                self.append(["nag_all"], [ns], mode="all")


            self.prm["l0"], self.prm["l1"] = 0, l0+l1

            flag, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, _, _, _, _ = self._run_sim(save_all=True, print_example=print_example)

            print_example = False
            if len(ns)==0:
                ns.append(np.nan)
            else:
                has_nag_data = True
            if len(n1s)==0: n1s.append(np.nan)

            if flag:
                self.append(["nag_seg", "fr_seg", "tr_seg", "m2max_seg", "nag_antag_seg", "m1max_seg", "mmax_seg"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="mean")
                self.append(["nstd_seg", "fstd_seg", "tstd_seg", "m2mstd_seg", "nstd_antag_seg", "m1mstd_seg", "mmstd_seg"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="std" )

                self.append(["nag_all_seg"], [ns], mode="all")

            if print_example:
                self.print2("first run done!")
            print_example = False
        ### run segaragated Ag


        return True

    def run_many(self):
        ### run many to get speed

        print_example=True

        has_nag_data = False

        for i in range(self.num_run):
            if not self.check("(run_many):"): ### check if there is any stop request
                return
            flag, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, m1tot, m2tot, fra, frb, eta_bar, eta_var = self._run_sim(save_all=True, print_example=print_example)
            print_example=False
            if len(ns)==0:
                ns.append(np.nan)
            else:
                has_nag_data = True
            if len(n1s)==0: n1s.append(np.nan)

            if flag:

                ntot = np.asarray(ns) + np.asarray(n1s)

                self.append(["nag", "fr", "tr", "m2max", "nag_antag", "m1max", "mmax", "ntot", "m1tot", "m2tot", "fra", "frb", "eta_bar", "eta_var"],
                            [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs, ntot, m1tot, m2tot, fra, frb, eta_bar, eta_var],
                            mode="mean")
                self.append(["nstd", "fstd", "tstd", "m2mstd", "nstd_antag", "m1mstd", "mmstd", "nstd_tot", "m1stdtot", "m2stdtot", "fstdra", "fstdrb", "etastd_bar", "etastd_var"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs, ntot, m1tot, m2tot, fra, frb, eta_bar, eta_var], mode="std" )
                #self.append(["tr_most_prob"], [trs], mode="most_prob")

                self.append(["tr_median"], [trs], mode="median")

                self.append(["tr_all"], [trs], mode="all")
                self.append(["nag_all"], [ns], mode="all")
                self.append(["m2max_all"], [m2maxs], mode="all")
        return has_nag_data

    def append(self, qtys, values, mode="mean"):
        for i, qty in enumerate(qtys):
            if mode == "mean":
                self.dataset.append(qty, np.nanmean(values[i]))
            elif mode =="std":
                self.dataset.append(qty, np.nanstd(values[i]))
            elif mode =="most_prob":
                self.dataset.append(qty, get_most_prob(values[i]))

            elif mode == "median":
                self.dataset.append(qty, get_median(values[i]))
            elif mode=="all":
                for vi in values:
                    self.dataset.append(qty, vi)
            else:
                self.dataset.append(qty, values[i])
        return

    def get_fidelity(self, random_l0=False):
        """
        simulate the selection fidelity
        """

        eb_distribution = [self.eb0, self.eb0+self.de]
        output=True
        for ni in range(self.num_run):
            nerror, terror, merror, kl = self.get_acc( eb_distribution, self.prm, self.cluster_num, output, True, random_l0=random_l0)

            self.append(["fidelity_n", "fidelity_t", "fidelity_m", "Dkl"],
                    [nerror, terror, merror, kl])
            output = False
            printProgress(ni, self.num_run)

        self.print2("DONE with fidelity! xi_n={0:.3f}, Dkl={1:.3f}".format(np.nanmean(self.dataset.get("fidelity_n")), np.nanmean(self.dataset.get("Dkl"))))
        return


    def perturb(self, prm_name, prm_value, prm_step, log_prm=False):
        '''
        perturb the system and get sensitivity
        '''

        self.print2("*"*15+"perturb "+ prm_name+"*"*15)

        ### init recording array
        sens, acc = np.zeros(self.ns*self.num_run),np.zeros(self.ns*self.num_run)
        log_sens = np.zeros(self.ns*self.num_run)

        n_ag1, n_ag2 =  np.zeros(self.ns*self.num_run),np.zeros(self.ns*self.num_run)
        n_std1, n_std2  = np.zeros(self.ns*self.num_run),np.zeros(self.ns*self.num_run)
        sens[:], acc[:], n_ag1[:], n_ag2[:],n_std1[:], n_std2[:] = np.nan, np.nan,np.nan, np.nan,np.nan, np.nan
        log_sens[:] = np.nan

        success_num = 0
        print_example= True
        self.print2("perturbing "+prm_name+" at " + str(prm_value) + " with step size " + str(prm_step))
        for num in range(self.ns*self.num_run):

            printProgress(num, self.ns*self.num_run)
            if not self.check("(perturb):"): ### check if there is any stop request
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            ### perturb the affinity
            self.sto.set_prm(prm_name, prm_value + prm_step)
            if print_example:
                self.print2("print example: prm1=" + str(self.sto.prm[prm_name]))

            finished1, nag1, nstd1 = self._run_sim(print_example=print_example)
            if not finished1:
                continue
            self.sto.set_prm(prm_name, prm_value)# - prm_step)

            if print_example:
                self.print2("print example: prm2=" + str(self.sto.prm[prm_name]))

            finished2, nag2, nstd2 = self._run_sim(print_example=print_example)

            if not finished2:
                continue
            success_num += 1
            print_example = False
            assert (not(np.isnan(nag1))) and (not(np.isnan(nag2)))

            sens[num] = ((nag1-nag2)/(2*prm_step) )
            acc[num] = sens[num]/np.sqrt( (nstd1**2 + nstd2**2)/2)
            n_ag1[num], n_ag2[num], n_std1[num], n_std2[num] = nag1, nag2, nstd1, nstd2
            if not log_prm:
                log_sens[num] =  ((np.log(nag1)- np.log(nag2))/(2*prm_step))
            else:
                log_sens[num] =  ((np.log(nag1)- np.log(nag2))/(np.log(prm_value + prm_step) - np.log(prm_value - prm_step) ))

            if success_num >= self.num_run:
                break


        self.sto.set_prm(prm_name, prm_value)
        if success_num < self.num_run:
            ### if the sucess_num is not enough
            ### we say B cell fail to extract the antigen
            self.print2("not enough sucessful attempts, success_num={0:d}".format(success_num), tag=False)
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        else:
            self.print2("-- nag1={0:.2f}[{2:.2f}], nag2={1:.2f}[{3:.2f}] (mean of {4:d} runs)".
                      format(np.nanmean(n_ag1), np.nanmean(n_ag2),
                             np.nanmean(n_std1), np.nanmean(n_std2),
                             success_num))
            self.print2("-- sens={0:.3f}[{1:.3f}], acc={2:.3f}[{3:.3f}], log sens={4:.3f}[{5:.3f}]".
                        format(np.nanmean(sens), np.nanstd(sens),
                               np.nanmean(acc), np.nanstd(acc),
                               np.nanmean(log_sens), np.nanstd(log_sens)))
        return n_ag1, n_ag2, n_std1, n_std2, sens, acc, log_sens

    def get_antag(self):
        print_example=True
        for i in range(self.num_run):
            self.sto.set_prm("l1", 0)
            if not self.check("(get_antag):"): ### check if there is any stop request
                return
            flag0, ns0, frs0, trs0, m2maxs0, n1s0, m1maxs0, mmaxs0, m1tot0, m2tot0, fra0, frb0, _, _ = self._run_sim(save_all=True, print_example=print_example)
            if not flag0:
                continue

            self.sto.set_prm("l1", self.l1)
            flag, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, m1tot, m2tot, fra, frb, _, _ = self._run_sim(save_all=True, print_example=print_example)

            if not flag:
                continue

            self.append(["nag0", "fr0", "tr0", "m2max0"], [ns0, frs0, trs0,m2maxs0], mode="mean")
            self.append(["nstd0", "fstd0", "tstd0", "m2mstd0"], [ns0, frs0, trs0,m2maxs0], mode="std" )

            self.append(["nag", "fr", "tr", "m2max", "nag_antag", "m1max", "mmax"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="mean")
            self.append(["nstd", "fstd", "tstd", "m2mstd", "nstd_antag", "m1mstd", "mmstd"], [ns, frs, trs,m2maxs, n1s, m1maxs, mmaxs], mode="std" )

            self.append(["m1tot0", "m2tot0", "fra0", "frb0"], [m1tot0, m2tot0, fra0, frb0], mode="mean")
            self.append(["m1tot", "m2tot", "fra", "frb"], [m1tot, m2tot, fra, frb], mode="mean" )

            self.append(["antagN"], [[(np.nanmean(ns0)-np.nanmean(ns))/np.nanmean(ns0)]], mode="mean")
            self.append(["antagT"], [[(np.nanmean(trs0)-np.nanmean(trs))/np.nanmean(trs0)]], mode="mean")
            print_example=False
        return


    def _run_sim(self, save_all=False, print_example=True): ### return flag, nag, nstd
        m2maxs, ns, frs, trs = [], [], [], []
        n1s, m1maxs = [], [] ## antagonist
        mmaxs = []
        if save_all:
            m1tots, m2tots = [], []
            fras, frbs = [], []
            eta_bars = []
            eta_vars = []

        for i in range(self.cluster_num):
            #sim.printProgress(i, self.cluster_num)
            self.sto.setup()
            flag, tr1, spec1, fr1 = self.sto.run1(output=False, storeData=False, check=self.check)
            if flag:
                ns.append(spec1[3])
                n1s.append(spec1[1])
                frs.append(fr1)
                trs.append(tr1/60000)
                m2maxs.append(self.sto.m2_max)
                m1maxs.append(self.sto.m1_max)
                mmaxs.append(self.sto.m_max)
                if save_all:
                    m1_tot, m2_tot = self.sto.get_mtot()
                    fra, frb = self.sto.get_mean_force_per_bond()
                    m1tots.append(m1_tot)
                    m2tots.append(m2_tot)
                    fras.append(fra) ## for ag 1
                    frbs.append(frb) ## for ag 2
                    eta_bars.append(self.sto.get_eta_moment(1))
                    eta_vars.append(self.sto.get_eta_moment(2))
            else:
                if save_all:
                    m2maxs.append(self.sto.m2_max)
                    m1maxs.append(self.sto.m1_max)
                    mmaxs.append(self.sto.m_max)
                    trs.append(np.inf)
                    frs.append(np.inf)

                    #ns.append(np.inf)
                    #n1s.append(np.inf)
                    #return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    return False, np.nan, np.nan  ### not all clusters are broken

            if print_example:
                if not flag:
                    self.print2("(_run_sim): un-finished!")
                self.print2("(_run_sim): nag={0:.2f},na={4:.2f}, fr={1:.2f}, tr={2:.2f}, mmax={3:.0f}, m2tot={5:.0f}".
                               format(self.sto.current_species[3], fr1/1000, tr1/60000, self.sto.m_max, self.sto.current_species[1], len(self.sto.fb_record[self.sto.fb_record>0])))
                print_example = False

            if not self.check("(_run_sim):"): ### check if there is any stop request
                if save_all:
                    return False, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, m1tots, m2tots, fras, frbs, eta_bars, eta_vars
                else:
                    return False, np.nan, np.nan

        if save_all:
            return True, ns, frs, trs, m2maxs, n1s, m1maxs, mmaxs, m1tots, m2tots, fras, frbs, eta_bars, eta_vars ### all cluster are broken
        else:
            nag, nstd = np.nanmean(ns), np.nanstd(ns)
            return True, nag, nstd ### all clusters are broken



    def get_acc(self, eb_list, prm, n=200, output=False, save_qty=False, random_l0=False):
        """
            args:
                eb_list: affinity distribution of B cells
                prm: default parameter (fixing other parameters)
                save_qty: save qty (i.e. nag, tr, fr, mm) for eb_list[0]
                n: number of trails
            return:
                acc: the probability to rank the best B cell highest
        """
        nacc_count = 0
        tacc_count = 0
        macc_count = 0

        best_id = np.argmax(eb_list)

        num_cell = len(eb_list)
        sto_list = []


        for i, eb in enumerate(eb_list):
            sto_list.append(sim.Stoch(copy.deepcopy(prm)))
            sto_list[i].set_prm("eb", eb)


        if output:
            self.print2("*"*15+"get_fidelity"+"*"*15)
            self.print2("eblist: "+", ".join(str(e) for e in eb_list))
            self.print2("best id={0:d}".format(best_id))
            self.print2("check eb: [", end="  ")
            for stoi in sto_list:
                self.print2("{0:.2f}".format(np.log(stoi.prm["tau_b"][1])), end=", ", tag=False)
            self.print2(" ]", tag=False)

            self.print2("check rates: [", end="  ")
            for stoi in sto_list:
                self.print2(",".join("{0:.2e}".format(r) for r in stoi.rates), end=", ", tag=False)
                self.print2("], [", tag=False, end="")
            self.print2(" ]", tag=False)
            self.print2("check init cond: [", end="")
            for stoi in sto_list:
                self.print2(",".join("{0:.0f}".format(mi) for mi in stoi.init_cond), end=", ", tag=False)
                self.print2("], [", tag=False, end="")
            self.print2("]", tag=False)


        if save_qty:
            m1maxs, m2maxs, n1s, ns, frs, trs = [], [], [], [], [], []

        nag_list_all = np.zeros((num_cell, n)) ### collecting all n_ags

        for i in range(n):
            nag_list = np.zeros(num_cell)
            tend_list = np.zeros(num_cell)
            mm_list = np.zeros(num_cell)

            if output and i>1:
                printProgress(i, n)
            for j, stoi in enumerate(sto_list):
                if random_l0:
                    stoi.set_prm("l1", self.sample_l0())
                    if output and i==1:
                        self.print2("check random l0 & l1: {0:d}, {1:d}".format(stoi.l0, stoi.l1))
                stoi.setup()
                flag, tr1, spec1, fr1  = stoi.run1(output=False, storeData=False)
                if not flag:
                    if output:
                        self.print2("simulation not finished!")
                    return np.nan, np.nan, np.nan, np.nan

                if j==0 and save_qty:
                    ns.append(spec1[3])
                    frs.append(fr1)
                    trs.append(tr1/60000)
                    m2maxs.append(stoi.m2_max)

                if j==1 and save_qty:
                    n1s.append(spec1[3])
                    m1maxs.append(stoi.m2_max)

                nag_list_all[j, i] = spec1[3]

                nag_list[j] = spec1[3]
                tend_list[j]= tr1
                mm_list[j] = stoi.m_max
            if best_id == np.random.choice(np.flatnonzero(nag_list == nag_list.max())): #np.argmax(nag_list):
                nacc_count += 1

            if best_id == np.argmax(tend_list):
                tacc_count += 1

            if best_id == np.argmax(mm_list):
                macc_count += 1

            if output and i==1:
                self.print2("example 1:")
                self.print2("\t nag_list=[" +",".join(str(nag) for nag in nag_list)+"]")
                self.print2("\t tendlist=["+",".join(str(tr/60000)[:5] for tr in tend_list)+"]")
                self.print2("\t mmaxlist=["+",".join(str(mm) for mm in mm_list)+"]")
        if output:
            self.print2("done!")
            self.print2("nacc_count={0:d}, tacc_count={1:d}, mmacc_count={2:d}".format(nacc_count,tacc_count, macc_count))
        if save_qty:
            self.append(["nag5", "nag6", "fr", "tr", "m2max"], [ns, n1s, frs, trs,m2maxs], mode="mean")
            self.append(["nstd5", "fstd", "tstd", "m2mstd"], [ns, frs, trs,m2maxs], mode="std" )
            self.append(["tr_most_prob"], [trs], mode="most_prob")

            self.append([ "m1max"], [m1maxs], mode="mean")
        kl = KL_plug_in(nag_list_all[0], nag_list_all[1])
        return nacc_count / n, tacc_count / n, macc_count/n, kl

    def get_acc_light(self, eb_list, prm, n=200, output=False):
        """
        light version of get acc, used for opt_prm
            args:
                eb_list: affinity distribution of B cells
                prm: default parameter (fixing other parameters)
                n: number of trails
            return:
                acc: the probability to rank the best B cell highest
        """
        nacc_count = 0
        best_id = np.argmax(eb_list)

        num_cell = len(eb_list)
        sto_list = []


        for i, eb in enumerate(eb_list):
            sto_list.append(sim.Stoch(copy.deepcopy(prm)))
            sto_list[i].set_prm("eb", eb)


        if output:
            self.print2("*"*15+"get_fidelity"+"*"*15)
            self.print2("eblist: "+", ".join(str(e) for e in eb_list), end=", ")
            self.print2("best id={0:d}".format(best_id))
            self.print2("check eb: [", end="  ")
            for stoi in sto_list:
                self.print2("{0:.2f}".format(np.log(stoi.prm["tau_b"][1])), end=", ", tag=False)
            self.print2(" ]", tag=False)

            self.print2("check init cond: [", end="")
            for stoi in sto_list:
                self.print2(",".join("{0:.0f}".format(mi) for mi in stoi.init_cond), end=", ", tag=False)
                self.print2("], [", tag=False, end="")
            self.print2("]", tag=False)

        un_broken_counts = 0

        for i in range(n):
            nag_list = np.zeros(num_cell)
            tend_list = np.zeros(num_cell)
            mm_list = np.zeros(num_cell)

            if output and i>1:
                printProgress(i, n)
            for j, stoi in enumerate(sto_list):
                stoi.setup()
                flag, tr1, spec1, fr1  = stoi.run1(output=False, storeData=False, check=self.check)
                if not flag:

                    if output:
                        self.print2("simulation not finished!")
                    un_broken_counts += 1
                    if i>20 and un_broken_counts/i > self.get_config("mini_surv_chance"):
                        self.print2("too many unbroken cluster. Skip to next prm! num_unbroken_cluster={0:d}, i={1:d}".format(un_broken_counts, i))
                        return 0
                    nag_list[j] = stoi.current_species[3]
                else:
                    nag_list[j] = spec1[3]
            if best_id == np.random.choice(np.flatnonzero(nag_list == nag_list.max())):
                nacc_count += 1

            if output and i==1:
                self.print2("example 1:")
                self.print2("\t nag_list=[" +",".join(str(nag) for nag in nag_list)+"]")

            if not self.check("(_get_acc_light):"):
                return 0

        if output:
            self.print2("done!")
            self.print2("nacc_count={0:d}".format(nacc_count))
        return nacc_count / n


    def simulated_annealing(self, objective, bounds, n_iterations, step_size, temp, prm_init=None, output=False):

        ## generate an initial point
        if prm_init is None:
            best = bounds[:, 0] + rd.rand(len(bounds))*(bounds[:, 1] - bounds[:, 0])
        else:
            best = prm_init

        self.print2("init prm= %s" % best)
        ### evaluate the initial condition
        best_eval = objective(best)

        ### current working solution
        curr, curr_eval = best, best_eval

        self.print2("> %d acc(%s) = %.5f" % (0, curr, -curr_eval))

        scores= []
        ## run the algorithm

        for i in range(n_iterations):
            ## take a step
            trials =0
            while True:
                candidate = curr +  rd.randn(len(bounds)) * step_size
                if np.all(candidate>bounds[:, 0]) and np.all(candidate<bounds[:, 1]):
                    break
                trials += 1
                if trials > 10:
                    step_size *= 2

            ### evauate candidate point
            candidate_eval = objective(candidate)
            if candidate_eval == 0: ### if the clusters are unbroken, we need to change f0
                if len(curr)==1 and self.opt_prm_names[0]=='f0':
                    bounds[0][0] = max(bounds[0][0], candidate[0])
                    self.print2("!!! update lower bound of f0, now bounds= %s" % bounds)


            if candidate_eval  < best_eval:
                best, best_eval = candidate, candidate_eval
                ## report the progress
                self.print2("ORZ updated optm %d acc(%s) = %.5f"%(i, best, -best_eval))
                scores.append(-best_eval)
            diff = candidate_eval - curr_eval

            t = temp / float(i + 1)

            ### calculate metropolis acceptance criterion
            metropolis = np.exp(-diff/t)

            if diff<=0 or rd.rand() < metropolis:
                curr, curr_eval = candidate, candidate_eval
                if output:
                    self.print2("> %d f(%s) = %.5f" % (i, curr, -curr_eval))

            if not self.check("(_simulated annealing):"):
                return [best, -best_eval], scores
        return [best, -best_eval], scores

    def gradient_descent(self, max_iterations, threshold, prm_init, obj_func, learning_rate=0.01, momentum=0.8, step_size=0.1, output=False):
        prm = prm_init
        prm_history = [prm]
        acc_history = [obj_func(prm)]
        delta_prm = step_size

        i = 0
        diff = 1.0e10

        if output:
            self.print2("[gradient_descent]:\n\t (i, prm, acc)=({0:d}, {1:.1f}, {2:.3f})".format(0, prm_history[-1], acc_history[-1]))

        while i<max_iterations and diff>threshold:
            prm = prm+ delta_prm
            new_acc = obj_func(prm)

            grad = (new_acc - acc_history[-1])/(prm - prm_history[-1])

            if i>0:
                delta_prm = -learning_rate*grad + momentum*delta_prm
            else:
                delta_prm = -learning_rate*grad
            prm_history.append(prm)
            acc_history.append(new_acc)



            i += 1
            diff = abs(acc_history[-1] - acc_history[-2])
            if output:
                self.print2("(i, prm, acc, delta_prm)=({0:d}, {1:.1f}, {2:.3f}, {3:.2f})".format(i, prm_history[-1], acc_history[-1], delta_prm))

        if output:
            if i==max_iterations:
                self.print2("max iteration reaches. Stop optimizing!")
            elif diff<threshold:
                self.print2("required threshold reaches, stop optimizing!")

        return prm_history, acc_history

    def find_optimal_gradient_descent(self, prm_name, prm_init, output=False):
        def obj_func(prm):
            self.sto.set_prm(prm_name, prm)
            eb_distribution = [self.eb0, self.eb0+self.de]

            nacc, tacc, macc, kl = self.get_acc( eb_distribution, self.prm, self.cluster_num, False, False, random_l0=False)
            return nacc
        return self.gradient_descent(10, 0.002, prm_init, obj_func, output=output, step_size=10)



    def opt_get_prm_init(self, noise=True):
        bounds = np.asarray(self.opt_prm_bounds)
        step_size = np.asarray(self.opt_step_size)
        prm_init= bounds[:, 0] + rd.rand(len(bounds))*(bounds[:, 1] - bounds[:, 0])

        if self.opt_prm_init == "bifurcation":
            fstar, mstar = getBifurPoint(self.prm)
            if 'f0' in self.opt_prm_names:
                prm_init[self.opt_prm_names.index('f0')] = fstar
            if 'mc' in self.opt_prm_names:
                prm_init[self.opt_prm_names.index('mc')] = mstar

        elif self.opt_prm_init == "bifur_slope":
            fstar, mstar = getBifurPoint(self.prm)
            if 'f0' in self.opt_prm_names:
                mc = self.prm["mc"]
                prm_init[self.opt_prm_names.index('f0')] = fstar*mc/mstar
            if 'mc' in self.opt_prm_names:
                f0 = self.prm["f0"]
                prm_init[self.opt_prm_names.index('mc')] = mstar*f0/fstar

        elif self.opt_prm_init == "close_gap":
            mc = get_ms(self.prm, False)
            fstar, mstar = getBifurPoint(self.prm)
            f0 = max(fstar, ((self.prm["eb"]-self.prm["ea"])*4.012/0.5)*mc)
            if 'f0' in self.opt_prm_names:
                prm_init[self.opt_prm_names.index('f0')] = f0
            if 'mc' in self.opt_prm_names:
                prm_init[self.opt_prm_names.index('mc')] = mc


        elif self.opt_prm_init == "last_best":
            prm_init = self.last_best

        self.print2("preparing init prm:, method=%s, value=%s" % (self.opt_prm_init, prm_init))
        if noise:
            prm_init = prm_init + rd.rand(*prm_init.shape)*step_size*0.2

        if np.all(prm_init>bounds[:, 0]) and np.all(prm_init<bounds[:, 1]):
            return prm_init
        else:
            self.print2("init prm outside bounds")
            return None


    def find_optimal(self, output=False):
        def obj_func(prms):
            for prmi, prm_name in zip(prms, self.opt_prm_names):
                self.sto.set_prm(prm_name, prmi)

                self.append(["opt_tried_"+prm_name], [prmi])
            eb_distribution = [self.eb0, self.eb0+self.de]
            nacc = self.get_acc_light( eb_distribution,
                                          self.sto.prm,
                                          self.cluster_num,
                                          output=False)
            self.append(["opt_acc_all"], [nacc])
            return -nacc

        bounds = np.asarray(self.opt_prm_bounds)
        step_size = np.asarray(self.opt_step_size)
        n_iterations = self.opt_n_iterations
        temp  = self.opt_temp
        prm_init = self.opt_get_prm_init(noise=True)

        self.print2("opt prm names = "+", ".join(self.opt_prm_names))
        self.print2("bounds %s" % bounds)
        self.print2("n_iterations %d" % n_iterations)
        self.print2("get prm init %s" % prm_init)
        self.print2("step size= %s" % step_size)
        self.print2("temparture %s" % temp)
        best, scores = self.simulated_annealing(obj_func, bounds, n_iterations,
                                                step_size, temp, prm_init, output=output)
        self.last_best = best[0]

        step_size = step_size/10
        self.print2("--- now reduce the step size to %s" % step_size)
        best, scores2 = self.simulated_annealing(obj_func, bounds, n_iterations,
                                                step_size, temp, best[0], output=output)

        if output:
            self.print2("score history: [" + ",".join(["{0:.3f}".format(si) for si in scores + scores2])+"]" )
        return best, scores





    #def _run_sim()
    def print_info(self):
        self.print2("-"*20 + "basic info" + "-"*20)

        self.print2("force info:  beta={0:.1f}, mc={1:.1f}, tc={2:.3f}, scheme={3:s}, f0={4:.3f}, tm={5:.1f}".
                   format(self.prm["beta"],  self.prm["mc"], self.prm["tc"]/60000,
                          self.prm["scheme"], self.prm["f0"]/1000, self.sto.tm/60000)
                   )
        self.print2("prm info:, ea={0:.1f}, eb={1:.2f},ec={2:.1f}, ed={3:.1f}, l0=[{4:d}, {5:d}], , eon={6:.2f}".
              format(self.prm["ea"], self.eb0, self.prm["ec"], self.prm["ed"], self.prm["l0"], self.prm["l1"],  self.prm["eon"]))

        self.print2("scan info: n_run={0:d}, n_clst={1:d}, ns={2:d}".format(self.num_run, self.cluster_num, self.ns))


        pass





from collections import Counter
def KL_plug_in(X, Y):
    x = Counter(X)
    y = Counter(Y)
    m, n = len(X), len(Y)

    kl = 0

    for xi, pi in x.items():
        if xi in y:
            pi, qi = x[xi]/m, y[xi]/n
        else:
            pi, qi = x[xi]/m, 1/n
        kl += pi*np.log(pi/qi)
    return kl
