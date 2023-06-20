import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

FIG_SIZE = (3, 2.6)
DPI = 150

clist = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def plot_dis_curve(scanner, qty="nag"):
    if qty=="nag":
        ax = plot_qty(scanner.dataset, qty_y = "nag", show=False, fmt0="-o")
        ax.set(xlabel="secondary antigen affinity", ylim=(0, 100), xlim=(7, 20))

        plot_qty(scanner.dataset, ax=ax, qty_y = "nag_antag", fmt0="-o", show=False)
        ax.plot(np.linspace(7, 23, 10), [scanner.dataset.get_mean("nag0")]*10, '--k')
        #ax.legend(["primary", "secondary"])
        ax.vlines(x=15.1, ymin=0, ymax=100, linestyle="dotted", color="gray")
    else:
        ax = plot_qty(scanner.dataset, qty_y = "tr", show=False, fmt0="-o")
        ax.set(xlabel="secondary antigen affinity", ylim=(0.1, 10),yscale="log", xlim=(7, 20))

        #plot.plot_qty(scanMc.dataset, ax=ax, qty_y = "nag_antag", fmt0="-o", show=False)
        ax.plot(np.linspace(7, 23, 10), [scanner.dataset.get_mean("tr0")]*10, '--k')
        #ax.legend(["primary", "secondary"])
        ax.vlines(x=15.1, ymin=0, ymax=100, linestyle="dotted", color="gray")
    plt.show()
    return

def plot_qty(data, qty_y, qty_x="prms", ax=None, show=True, errorbar=False, qty_y_err=None,qty_x_err=None, fill_between=False,fmt0='o', **keyargs):
    ax = get_ax(ax, xlabel=qty_x, ylabel=qty_y)
    xdata, ydata = data.get(qty_x), data.get(qty_y, array=True)
    if errorbar:
        if qty_y_err is None:
            yerr = data.get(qty_y + "_std", array=True)
        else:
            yerr = data.get(qty_y_err, array=True)

        if qty_x_err is None:
            xerr = data.get(qty_x + "_std", array=True)
        else:
            xerr = data.get(qty_x_err, array=True)

        if qty_x == "prms":
            ax.errorbar(xdata, ydata, yerr=yerr, fmt=fmt0, **keyargs)
        else:
            ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt=fmt0, **keyargs)
        # ax.plot(xdata, ydata, fmt0, ms=4)
    else:
        ax.plot(xdata, ydata, fmt0, ms=4, **keyargs)
    return ret_ax(ax, show)


### plot sto traj
def plot_sto_traj(sto, ax=None, show=True, plotAg=True,indexs=[2, 3], **keyargs):
    ax = get_ax(ax, "time", "# of complex")
    for index in indexs:
        ax.plot(sto.t_record/60000, [spec[index] for spec in sto.spec_record], **keyargs)
    return ret_ax(ax, show)


### plot traj in phase spaxe
def plot_sto_traj_phase(sto=None, ax=None, show=True, bifur=True, scale=1, sep=800, **keyargs):
    ax = get_ax(ax, "force", "# of complex")
    if sto is None:
        pass
    else:
        #ax.plot(sto.f_record/scale, [spec[2]/scale for spec in sto.spec_record], **keyargs)
        ax.plot(np.asarray(sto.history["f"])/scale, np.asarray(sto.history["m3"])/scale, **keyargs)
    if bifur:
        #sep = 800
        x, f, y = getBifur(sto.prm)
        ax.plot(f[:sep]/scale, x[:sep]/scale, '--k', linewidth=1)
        ax.plot(f[sep:]/scale, x[sep:]/scale, '-k', linewidth=1)
    return ret_ax(ax, show)


def plot_bifur_curve(sto=None, ax=None, sep=800,scale=1,  **keyargs):
    ax = get_ax(ax, "force", "# of complex")
    x, f, y = getBifur(sto.prm)
    ax.plot(f[:sep]/scale, x[:sep]/scale, '--', linewidth=1, **keyargs)
    ax.plot(f[sep:]/scale, x[sep:]/scale, '-', linewidth=1, **keyargs)
    return ax



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


def get_ax(ax, xlabel, ylabel):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax

def ret_ax(ax, show):
    if show:
        plt.show()
        return
    else:
        return ax

