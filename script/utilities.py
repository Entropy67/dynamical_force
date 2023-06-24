'''

useful functions


'''

import numpy as np
import json
from mpmath import mp
from collections import Counter
from statistics import median
from scipy import optimize as opt


### get bifur curve
def getBifur(prm):
    kT = 4.012
    N = 1000
    xlist = np.concatenate([np.linspace(0, 0.99, N), np.linspace(0.99, 1, N)])
    kon, k10, k20 = 1/prm["ton"], 1/prm["tau_a"][1], 1/prm["tau_b"][1]
    K1, K2 = k10/(kon), k20/(kon)
    f1, f2 = kT/prm["xb1"], kT/prm["xb2"]
    l0 = prm["l1"]
    flist, ylist = np.zeros_like(xlist),np.zeros_like(xlist)
    for i, xi in enumerate(xlist):
        func = lambda f: (K1*np.exp(f/(xi*f1)) + K2*np.exp(f/(xi*f2)))*xi-(1-xi)
        f0 = min(xi*np.log((1-xi)/(K1*xi))*f1, xi*np.log((1-xi)/(K2*xi))*f2)
        fi = opt.fsolve(func, f0)
        if fi>0:
            flist[i] = fi
            ylist[i] = xi*K1*np.exp(fi/(xi*f1))
        else:
            flist[i] = None
            ylist[i] = None
    return xlist*l0, flist*l0, ylist*l0

### get bifur point

def getBifurPoint(prm):
    x, f, y = getBifur(prm)
    return np.nanmax(f), x[np.nanargmax(f)]

def get_ms(prm, use_mc=True):
    if use_mc:
        if prm["scheme"] in ["const", "hillN"]:
            return prm["mc"]
    kon, k10, k20 = 1/prm["ton"], 1/prm["tau_a"][1], 1/prm["tau_b"][1]
    return kon*prm["l1"]/(kon+k10+k20)



def get_median(array):
    return median(array)

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

def sort_dict_by_qty(my_dict, qty, reverse=False):
    if not qty in my_dict:
        print("sort_dict_by_qty: error, no such qty in my_dict")
        return

    n = len(my_dict[qty])

    def switch(i, j):
        for k in my_dict.keys():
            tmp = my_dict[k][i]
            my_dict[k][i] = my_dict[k][j]
            my_dict[k][j] = my_dict[k][i]
        return

    for i in range(n):
        for j in range(i+1, n):
            if reverse:
                if my_dict[qty][i] < my_dict[qty][j]:
                    switch(i, j)
            else:
                if my_dict[qty][i] > my_dict[qty][j]:
                    switch(i, j)
    return


def convert_speed(datai, qty="tr"):
    tr = datai.get(qty, array=True)
    tr_std =datai.get("tr_std", array=True)
    vr = 1/tr
    vr_std = tr_std/tr**2

    if "vr" in datai.dataset.keys():
        datai.dataset["vr"] = []
        datai.dataset["vr_std"] = []
    for i, vi in enumerate(vr):
        datai.append(["vr", "vr_std"], [vi, vr_std[i]])
    return



def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def get_pareto_front(sci, qtyx="fidelity_n", qtyy="vr"):

    c1 = sci.get(qtyx, True)
    c2 = sci.get(qtyy, True)

    n = c1.shape[0]

    costs = np.zeros((n, 2))
    costslog = np.zeros((n, 2))
    costslog[:, 0] = c1
    costslog[:, 1] = np.log(c2)
    for i in range(n):
        costs[i][0] = c1[i]
        costs[i][1] = c2[i]
    mask = identify_pareto(costslog)
    return costs, costs[mask], mask


def sync_prm(prm):
    prm["tau_a"] = [float(np.exp(prm["ec"])), float(np.exp(prm["ea"]))]
    prm["tau_b"] = [float(np.exp(prm["ed"])), float(np.exp(prm["eb"]))]
    prm["ton"] = float(np.exp(prm["eon"]))
    if "vc" in prm:
        prm["tc"] = 60000.0/prm["vc"]

    if "logtc" in prm:
        prm["tc"] = 60000.0*(10.0**(prm["logtc"]))
    return prm

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