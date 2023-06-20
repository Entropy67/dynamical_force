'''
general scaner to scan parameters

'''

import numpy as np
import json
import os.path

import .interface as intface
import .controller as controller
from .utilities import *


dumn_agent = controller.Controller()

class Scaner(controller.Controller):

    def __init__(self, agent=dumn_agent, double_prm=False, random_sample=False):

        self.double_prm = double_prm ### simultaneously vary two parameters
        self.random_sample = random_sample ### random sample parameters
        super().__init__()
        self.saved_qty = ["prms", "prms2"]
        self.load_agent(agent)

        self.update_config({
            "prm_name": "f0",
            "prm_list_prm": [1, 10, 1],
            "unit":1000,
            "prm_scale": "linear",

            "prm2_name": "tc",
            "prm2_list_prm": [0.1, 1, 10],
            "unit2": 60000,
            "prm2_scale": "linear"
        })


        self.tag = "Scaner"
        self.color = "green"
        self.last_run = 0 ### for continue run
        self.early_stop = False ### for double prm only. When getting no Ag, skip prm2
        self.build()
        pass

    def load_agent(self,agent):
        self.agent=agent
        for qty in agent.saved_qty:
            self.saved_qty.append(qty)
            self.saved_qty.append(qty+"_std")
        return



    def build(self):
        super().build()

        #self.agent.load_config(self.config)
        self.prm_name = self.get_config("prm_name")
        self.prm_list = self._get_prm_list(self.get_config("prm_list_prm"), self.get_config("prm_scale"))
        self.unit = self.get_config("unit")
        try:
            self.prm2_name = self.get_config("prm2_name")
            self.prm2_list = self._get_prm_list(self.get_config("prm2_list_prm"), self.get_config("prm2_scale"))
            self.unit2 = self.get_config("unit2")
        except:
            self.prm2_name = ""
            self.prm2_list = []
            self.unit2 = 1
        self.agent_file = self.get_config("dir")+"/agent_data"
        return

    def run(self, end=99999):
        '''
        end is used to check the simulation at early stage;
        when the number of prm tried exceeds end, the simulation will end
        then you can try to call run function again and the simulation will continue
        '''
        self.print2("...loaded the job parameters successfully. ")
        self.print2("...PID:"+str(os.getpid()))
        self.print2("start running ........")
        if self.write_log:
            self.print2(">>write to log:"+self.log_file)
        self.print2(">>temp saved data:" + self.temp_data_file)
        if self.save:
            self.print2(">>final saved data:" +self.data_file)
            self.print2(">>summary saved:" +self.summary_file)
            self.print2(">>agent data saved to:" +self.agent_file)
            self.summary(info='\t'.join(qty for qty in self.summary_qty))

        self.print2(">> prm_name={0:s}, prm_list=".format(self.prm_name)+"["+",".join('{0:.3f}'.format(scan_prm_i) for scan_prm_i in self.prm_list)+']' )
        if self.double_prm:
            self.print2(">> prm2_name={0:s}, prm2_list=".format(self.prm2_name)+"["+",".join('{0:.3f}'.format(scan_prm_i) for scan_prm_i in self.prm2_list)+']' )
        #self.dataset.append("prm_name", self.prm_name)
        i0 = self.last_run
        for i, prm in enumerate(self.prm_list[i0:]):

            if not self.check("run"): ### check if there is any stop request
                return
            if i>end: ### terminate the simulation manually to check the results
                return
            if not self.output:
                printProgress(i+i0, len(self.prm_list))
            if self.double_prm:
                for j, prm2 in enumerate(self.prm2_list):
                    if not self.check("run"): return
                    finished = self._run(prm, prm2)
                    if self.early_stop and not finished:

                        break
            else:

                prm2 = self.constrain(prm)
                self._run(prm, prm2)
            self.last_run += 1

        if not self.output:
            printProgress(len(self.prm_list), len(self.prm_list))
        self.print2("calculation finished!!")
        self.print2("*"*40)
        self.last_run = 0
        return


    def _run(self, prm, prm2=None):
        self.print2("*"*20+" now on "+self.prm_name+"={0:.3f}".format(prm)+"*"*20)
        self.agent.set_prm(self.prm_name, prm*self.unit)

        if prm2 is not None:
            self.agent.set_prm(self.prm2_name, prm2*self.unit2)
            self.print2("*"*20+" now on "+self.prm2_name+"={0:.3f}".format(prm2)+"*"*20)
            ##self.agent.set_prm(self.prm2_name, prm2*self.unit2)

        finished = self.agent.run()

        data_temp = self.agent.get_data()
        self.append(data_temp)
        self.dataset.append("prms", prm)
        if self.double_prm:
            self.dataset.append("prms2", prm2)
        if len(self.summary_qty)>0:
            if self.double_prm:
                self.print2(">>>> sum: prm={0:.2f}, prm2={1:.2f}".format(prm, prm2))
            else:
                self.print2(">>>> sum: prm={0:.2f}".format(prm))
            info = ">>>>    : "
            for j, qty in enumerate(self.summary_qty):
                if (j+1)%5==0:
                    info += "\n>>>>    : "
                if qty == "prms" or qty=="prms2":
                    continue

                info += qty + "={0:.3f}[{1:.3f}], ".format(data_temp.get_mean(qty), data_temp.get_std(qty))
            self.print2(info)

        if self.save:
            self.save_to_temp()  ### temp save data
            self.agent.dataset.dump_to_txt(self.agent_file, unique=False, mod='a', drop_zeros=True, title="prm="+str(prm)) ### save agent data
            info = ""
            for qty in self.summary_qty:
                if qty == "prms":
                    info += "{0:.2f}\t".format(prm)
                    continue
                if qty == "prms2" and self.double_prm:
                    info += "{0:.2f}\t".format(prm2)
                    continue
                info += "{0:.3f}[{1:.3f}]\t".format(data_temp.get_mean(qty), data_temp.get_std(qty))
            self.summary(info=info)
        return finished

    def append(self, dataset):
        def append_all(qty):
            try:
                mean, std = dataset.get_mean(qty), dataset.get_std(qty)
            except:
                mean, std = [np.nan], [np.nan]
            if np.isnan(mean).any():
                self.dataset.append(qty, np.nan)
                self.dataset.append(qty+"_std", np.nan)
            else:
                self.dataset.append(qty, mean)
                self.dataset.append(qty+"_std", std)


        for i, qty in enumerate(self.saved_qty):
            if False and len(self.dataset.get("prms"))!=len(self.dataset.get(qty)):
                print("abundant or missing data point")
                print("len of prms:", len(self.dataset.get("prms")), ", len of qtys:", len(self.dataset.get(qty)))
                print("prms:", self.dataset.get("prms"))
                print("i=", i, "," , qty, ":", self.dataset.get(qty))
                raise Exception("error")
            if qty !="prms" and qty !="prms2" and qty != "prm_name" and qty[-4:] != "_std":
                append_all(qty)
        return


    def set_prm(self, prm_name, prm_value):
        self.build()
        self.agent.set_prm(prm_name, prm_value)
        return

    def _get_prm_list(self, prm_list_prm, scale="linear"):
        if prm_list_prm is None:
            return []
        if len(prm_list_prm)==3:
            if self.random_sample:
                print(prm_list_prm)
                prm_list = np.random.uniform(low=prm_list_prm[0], high=prm_list_prm[1], size=prm_list_prm[2])
            else:
                prm_list = np.arange(prm_list_prm[0], prm_list_prm[1], prm_list_prm[2])
        else:
            prm_list = np.asarray(prm_list_prm)

        if scale=="log":
            prm_list = np.exp(prm_list)
        elif scale=="log10":
            prm_list = 10**prm_list
        return prm_list.tolist()


    def constrain(self,prm1):
        #####return None
        """
        this function impose a constrain between prm1 and prm2
        default, None
        """
        if False:
            self.print2("constrain function is called. prm2 = {0:.1f}".format(20-prm1))
            return 25 - prm1
        return None



