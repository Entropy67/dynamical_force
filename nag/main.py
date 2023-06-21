
### run the function
import logging
import sys
import argparse
sys.path.insert(-1, "../script/")

import prm_scan as scan
from utilities import *
import interface as intf



def main():
    prm = load_prm("param.json")
    sync_prm(prm=prm)
    config = load_prm("config.json")
    logging.basicConfig(filename=config["dir"]+'/log',level=logging.DEBUG, format='%(asctime)s %(message)s')

    agent = intf.Agent(prm, config["jobs"])
    agent.load_config(config)
    agent.set_config("save", False)
    agent.get_info()


    scaner = scan.Scaner(agent, random_sample=False, double_prm=config["double_prm"])
    scaner.early_stop = False
    scaner.load_config(config)
    scaner.summary_qty = config["summary_qty"]
    scaner.get_info()

    scaner.run()
    scaner.close()
    agent.close()
    return

if __name__ == "__main__":
    main()
