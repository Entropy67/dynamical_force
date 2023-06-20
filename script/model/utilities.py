'''

useful functions


'''

import numpy as np
import json
from mpmath import mp

def get_most_prob(array):
    hist, bins = np.histogram(array, bins=30)
    index = np.argmax(hist)
    return (bins[index] +bins[index+1])/2

def set_eb(eb, prm):
    prm["tau_b"][1] = float(np.exp(eb))
    return

def set_ea(ea, prm):
    prm["tau_a"][1] = float(np.exp(ea))
    return


def sync_prm(prm):
    set_ea(prm["ea"], prm)
    set_eb(prm["eb"], prm)
    prm["tau_a"][0] = float(np.exp(prm["ec"]))
    prm["tau_b"][0]  = float(np.exp(prm["ed"]))
    prm["ton"] = float(np.exp(prm["eon"]))
    return prm

def check_prm(prm):
    assert prm["tau_b"][1] == float(100*np.exp(prm["eb"])), print("!!!! Error: prm is not synced")
    assert prm["tau_a"][1] == float(100*np.exp(prm["ea"])), print("!!!! Error: prm is not synced")
    return

def load_prm(filename):
    ### load the json
    with open(filename, 'r') as fp:
        prm = json.load(fp)
    return prm

def dump_prm(filename, prm):
    pass

def convert_dict_to_string(my_dict):
    info = "{\n"
    for key, value in my_dict.items():
        info += "\t"
        info += key
        info += ": " + str(value)
        info += "\n"
    info += "\n}"
    return info

def gen_pi(n):
    if n==0:
        return '_3'
    elif n==1:
        return '.1'
    else:
        return str(mp.pi)[n+1]


def printProgress(n, N):
    percent = int(100.0*n/N)
    toPrint = "progress: "
    for i in range(percent//5):
        toPrint += '|'
    toPrint += "{:d}%    ".format(percent)
    print(toPrint, end='\r')
    return