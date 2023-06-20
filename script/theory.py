import numpy as np
from scipy import optimize as opt
from scipy.special import binom

tUnit = 60000


##############################
### lifetime starting from all-closed state under constant force


kT = 4.14
x1 = 1.5
x2 = 2.0

def eta(ta, tb, f):
    return tb/(ta*np.exp(-f*(x1-x2)/kT)+tb)

def tau(ta, tb,f):
    return ta*np.exp(-f*x1/kT)*eta(ta, tb, f)

def rm(i, ta, tb, f):
    return i*(np.exp(f*x1/(i*kT))/ta + np.exp(f*x2/(i*kT))/tb)

def drm(i, ta, tb, f):
    return  i*(np.exp(f*x2/(i*kT))/tb)/rm(i, ta, tb, f)**2

def gm(i, n0, ton):
    return (n0-i)/ton


def lifetime_ana(n0, ta, tb, F, ton=0):
    ave = 0
    for i in range(1, n0+1):
        ave += 1.0/rm(i, ta, tb, F)
    ave0 = ave
    for i in range(1, n0):
        for j in range(i+1, n0+1):
            num, denom = 1, 1
            if True:
                for k in range(i, j):
                    num = num*gm(k, n0, ton)
                for k in range(i, j+1):
                    denom = denom*rm(k, ta, tb, F)
            #print(ave)
            ave += num/denom
    return ave


def lifetime_sens(n0, ta, tb, F, ton, dt=1.1):
    t1 = lifetime_ana(n0, ta, tb, F, ton)
    t2 =lifetime_ana(n0, ta, tb*1.1, F,ton)
    return (np.log(t2)-np.log(t1))/(np.log(1.1))

def lifetime_sens_ana(n0,ta,tb,F,ton):
    sens= 0
    for i in range(1, n0+1):
        sens += drm(i, ta, tb, F)

    for i in range(1, n0):
        for j in range(i+1, n0+1):
            num, denom = 1, 1
            if j-i>=j:
                num = 0
            for k in range(j-i, j):
                num = num*gm(k, n0, ton)
                flag=True

            for k in range(j-i, j+1):
                denom = denom*rm(k, ta, tb, F)

            dev = 0
            for k in range(j-i, j+1):
                dev += (1- eta(ta, tb, F/k))/k
            #print(ave)
            sens += num*dev/denom

    return sens/lifetime_ana(n0, ta, tb, F, ton)


def lifetime_sens_ana2(n0,ta,tb,F,ton):
    sens= 0
    for i in range(1, n0+1):
        sens += drm(i, ta, tb, F)

    for i in range(1, n0):
        for j in range(i+1, n0+1):
            num, denom, dev = 1, 1, 0

            for k in range(i, j):
                num *= gm(k, n0, ton)

            for k in range(i, j+1):
                denom *= rm(k, ta, tb, F)

            for k in range(i, j+1):
                dev += rm(k, ta, tb, F)*drm(k, ta, tb, F)
            #print(ave)
            sens += num*dev/denom

    return sens

def lifetime_std_ana(n0, ta, tb, F, ton):
    ### calculate product gm i, j
    phi = np.zeros((n0+2, n0+2))

    for i in range(1, n0+1):
        for j in range(i+1, n0+1):
            num, denom = 1, 1
            if i>= j:
                num =0
            for k in range(i, j):
                num =num*gm(k, n0, ton)
            for k in range(i, j+1):
                denom = denom*rm(k, ta, tb, F)
            phi[i, j] = num/denom

    ## find the mean lifetime list
    T_ave_list = np.zeros(n0+1)
    for n in range(1, n0+1):
        t_ave = 0
        for i in range(1, n+1):
            t_ave += 1.0/rm(i, ta, tb, F)

        for i in range(1, n+1):
            for j in range(i+1, n0+1):
                t_ave += phi[i, j]

        T_ave_list[n] = t_ave

    #print(T_ave_list)
    #print("lifetime=", T_ave_list[n0])
    assert abs(T_ave_list[-1]-lifetime_ana(n0, ta, tb, F, ton))/T_ave_list[-1]<0.000001, "lifetime doesn't match: {0:.3e} vs {1:.3e}".format(T_ave_list[-1], lifetime_ana(n0, ta, tb, F, ton))
    ### calculate the variance
    T_std = 0
    for i in range(1, n0+1):
        T_std += 2*T_ave_list[i]/rm(i, ta, tb, F)

        for j in range(i+1, n0+1):
            T_std +=2*T_ave_list[j]*phi[i,j]
    #print("std =", np.sqrt(T_std-T_ave_list[n0]**2)/T_ave_list[n0])
    return np.sqrt(T_std-T_ave_list[n0]**2)

def lifetime_xi(n0, ta, tb, F, ton, max_t=-1):
    if max_t <0:
        return (lifetime_sens_ana2(n0, ta, tb, F, ton)/lifetime_std_ana(n0, ta, tb, F, ton))**2

    elif lifetime_ana(n0, ta, tb, F, ton)<max_t:
        return (lifetime_sens_ana2(n0, ta, tb, F, ton)/lifetime_std_ana(n0, ta, tb, F, ton))**2, np.nan
    else:
        return np.nan, (lifetime_sens_ana2(n0, ta, tb, F, ton)/lifetime_std_ana(n0, ta, tb, F, ton))**2




###############################
## rupture from steady state
def nag1(prm):
    mBifur, nBifur, fBifur = bifur_state(prm)
    return nBifur+extract_ag(prm, fBifur, mBifur )

def nag1_std(prm):
    mBifur, nBifur, fBifur = bifur_state(prm)
    sigma_m_r, sigma_n_r = bifur_state_std(prm)
    eta = extract_ag(prm, fBifur, mBifur )/mBifur
    sigma_ext = extract_ag_std(prm, fBifur, mBifur)
    #var = sigma_n_r**2 + mBifur**2*sigma_eta**2 +sigma_m_r**2 * eta +sigma_m_r**2*sigma_eta**2
    var = sigma_n_r**2 + sigma_ext**2
    return sigma_n_r, sigma_ext, np.sqrt(var)


###############################
def bifur_state(prm):
    #### return bifurcation state
    xlist, flist, ylist = getBifur(prm)
    index = np.nanargmax(flist)
    #print(index)
    mBifur, nBifur, fBifur = xlist[index], ylist[index], flist[index]
    return mBifur, nBifur, fBifur

def bifur_state_std(prm):
    #### return bifurcation state
    xlist, flist, ylist = getBifur(prm)
    index = np.nanargmax(flist)
    #print(index)

    mBifur, nBifur, fBifur = xlist[index], ylist[index], flist[index]

    prmS = read_prm(prm, 0)
    l0 = prm["l0"]
    ### get rate
    fn = fBifur/mBifur
    kon = 1/prm["ton"]
    ka, kb = prmS["k10"]*np.exp(fn/prmS["f1"]), prmS["k20"]*np.exp(fn/prmS["f2"])

    ##
    sigma_m = np.sqrt(kon*l0*(ka+kb))/(ka+kb+kon)
    sigma_n = np.sqrt(ka*l0*(ka+kb))/(ka+kb+kon)
    return sigma_m, sigma_n

### get bifur curve
def getBifur(prm):
    kT = 4.012
    N = 1000
    xlist = np.concatenate([np.linspace(0,1, N)])
    kon, k10, k20 = 1/prm["ton"], 1/prm["tau_a"][1], 1/prm["tau_b"][1]
    K1, K2 = k10/(kon), k20/(kon)
    f1, f2 = kT/prm["xb1"], kT/prm["xb2"]
    l0 = prm["l0"]
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


def steady_state(prm, f=0):
    ## ss under const F
    ## only the first moment considered in expansion
    ## equilivalent to deterministic equation
    prmS = read_prm(prm, f)
    return prmS["ms"], prmS["ns"]

def steady_state_std(prm, f=0):
    ### return the stdev of m and n near steady state
    ### constant force
    ## only the first momnet is considered

    prmS = read_prm(prm, f)
    ms, ns = steady_state(prm ,f)
    l0 = prm["l0"]

    ### get rate
    fn = f/ms
    kon = 1/prm["ton"]
    ka, kb = prmS["k10"]*np.exp(fn/prmS["f1"]), prmS["k20"]*np.exp(fn/prmS["f2"])

    ##
    sigma_m = np.sqrt(kon*l0*(ka+kb))/(ka+kb+kon)
    sigma_n = np.sqrt(ka*l0*(ka+kb))/(ka+kb+kon)
    return sigma_m, sigma_n


def dynamics_constF(prm, t, f=0):
    t = t*tUnit
    prmS = read_prm(prm, f=f)
    ms, ns = prmS["ms"], prmS["ns"]
    mt = ms*(1-np.exp(-prmS["a"]*t))
    nt = ns*(1-(prmS["a"]*np.exp(-prmS["kon"]*t) - prmS["kon"]*np.exp(-prmS["a"]*t))/(prmS["a"]-prmS["kon"]))
    return mt, nt

def extract_ag(prm, f, m0):
    prmS = read_prm(prm, f=f)
    nag=0
    for i in range(int(m0), 0,-1):
        fn = f/i
        #print(i, fn)
        ka, kb = prmS["k10"]*np.exp(fn/prmS["f1"]), prmS["k20"]*np.exp(fn/prmS["f2"])
        if np.isnan(ka) or np.isnan(kb) or np.isinf(ka) or np.isinf(kb):
            continue
        nag += eta(ka, kb)
        #print(i, ka, kb, nag)
    fn = f/m0
    ka, kb = prmS["k10"]*np.exp(fn/prmS["f1"]), prmS["k20"]*np.exp(fn/prmS["f2"])
    if np.isnan(ka) or np.isnan(kb) or np.isinf(ka) or np.isinf(kb):
        pass
    else:
        nag += (eta(ka, kb)*(m0-int(m0)))
    return nag

def extract_ag_std(prm, f, m0):
    prmS = read_prm(prm, f=f)
    nvar=0
    for i in range(int(m0), 0,-1):
        fn = f/i
        #print(i, fn)
        ka, kb = prmS["k10"]*np.exp(fn/prmS["f1"]), prmS["k20"]*np.exp(fn/prmS["f2"])
        if np.isnan(ka) or np.isnan(kb) or np.isinf(ka) or np.isinf(kb):
            continue
        nvar += eta(ka, kb)*(1-eta(ka, kb))
    return np.sqrt(nvar)


def eta(ka, kb):
    return 1.0/(1.0+kb/ka)

def read_prm(prm, f=0):
    kT = 4.012
    kon, k10, k20 = 1/prm["ton"], 1/prm["tau_a"][1], 1/prm["tau_b"][1]
    K1, K2 = k10/kon, k20/kon
    f1, f2 = kT/prm["xb1"], kT/prm["xb2"]
    try:
        l0 = prm["l0_list"][1]
    except:
        l0 = prm['l0']
    a = (kon+k10*np.exp(f/f1)+k20*np.exp(f/f2))
    ms = l0*kon/a
    ns = k10*np.exp(f/f1)*ms/kon

    prmS = {
        "kon": kon,
        "k10": k10,
        "k20": k20,
        "K1": K1,
        "K2": K2,
        "f1":f1,
        "f2":f2,
        "a": a,
        "ms": ms,
        "ns": ns
    }
    return prmS



