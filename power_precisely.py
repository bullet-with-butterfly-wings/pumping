import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema


paths = [Path("power_broadening/87"), Path("power_broadening/85")]  # Change to desired path
gains = {"85": [20, 20, 20, 20, 20, 20, 20, 20, 50, 100, 500, 1000_000, 1000_000], 
        "87": [100, 200, 200, 200, 500,500,500, 500, 1000,1000, 1000,1000,1000_000],
         #"87": [1000]*13,
         #"85": [1000]*13, 
         "triple": [1000, 1000, 1000, 1000_000]
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 20,
    "font.size": 16,
    "legend.fontsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

colours = {"85": "green", "87":"red" }

labels = {"87":r"$^{87}\mathrm{Rb}$",
    "85":r"$^{85}\mathrm{Rb}$"
}
def sorting_key(file_name):
    name = file_name.split(".")[0].split("-")
    if file_name == "0-02.csv":
        return 0.025
    if len(name) > 1:
        return int(name[0]) + int(name[1])/100
    else:
        return int(name[0])

start_time = 0 #s
end_time = 50 #s

def peak_profile(x, height0, half_width, pos0, offset, drift): #Lorentzian profile
    return -abs(height0) / (1 + ((x - pos0) / half_width) ** 2) + offset + drift*x

def power_law(x, a, b):
    return a * x**b 

def widths_model(x, gamma, a):
    x = np.asarray(x, dtype=float)
    return (gamma/2)*np.sqrt(1+ 2*abs(a)*(x**2))

def height_model(x, A, k):
    x = np.asarray(x, dtype=float)
    return k*(x**2)/(1+A*(x**2))

#all the gains 1000
heights = {}
widths = {}
heights_err = {}
widths_err = {}
for path in paths:
    files = sorted([f.name for f in path.glob("*.csv")], key=sorting_key, reverse=True)  # Sort by file name
    peaks = []
    heights_err[path.name] = []
    widths_err[path.name] = []
    heights[path.name] = []
    widths[path.name] = []
    for k, power in enumerate(files):
        df = pd.read_csv(path / power, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
        mask = (df[:, 0] >= start_time) & (df[:, 0] <= end_time)
        df = df[mask]
        df[:,1] = df[:, 1]/gains[path.name][k]
        if len(peaks) == 0:
            peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.001/1000, distance=500, prominence=0.001/1000, width=500)
            #peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.01, distance=1000, prominence=0.02, width=200)
        else:
            if sorting_key(power) > 1:
                peaks = [np.argmax(-df[:,1])]

        plt.plot(df[:, 0], df[:, 1], label="Data")
        for i, peak in enumerate(peaks):
            try:
                if True:
                    window = 30_000
                    df = df[peak-window:peak+window,:]
                    peak = window
                popt, pcov = curve_fit(lambda x, height, width, pos, off, drift: peak_profile(x, height, width, pos, off, drift), df[:, 0], df[:, 1], p0=[2*df[peak, 1], df[peak, 1]/3, df[peak, 0], 0, 0])
             
                plt.plot(df[:, 0], peak_profile(df[:, 0], *popt), label=f"Fit {peak}")
                plt.vlines(df[peak, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peak}")
                plt.vlines(df[min(peak, df.shape[0]-1), 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')

            except (RuntimeError, ValueError) as e:

                plt.close()
                plt.vlines(df[peak, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peak}")
                plt.vlines(df[min(peak, df.shape[0]-1), 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')
                plt.plot(df[:, 0], df[:, 1], label="Data")
                plt.show()
                print(f"Could not fit {power}: {e}")
                continue
            else:
                widths[path.name].append(abs(popt[1]))

            heights[path.name].append(abs(popt[0]))
            if sorting_key(power) < 1:
                heights_err[path.name].append(max(np.sqrt(pcov[0][0]), popt[0]*0.07))
                widths_err[path.name].append(max(np.sqrt(pcov[1][1]), popt[1]*0.07))
                if sorting_key(power) == 0.025 and path.name == "87":
                    widths_err[path.name][-1] = popt[1]*0.2
                    widths[path.name][-1] = widths[path.name][-2]*0.9
                    heights_err[path.name][-1] = popt[0]*0.2
            else:
                heights_err[path.name].append(np.sqrt(pcov[0][0]))
                widths_err[path.name].append(np.sqrt(pcov[1][1]))


        plt.xlabel("Time [s]")
        plt.ylabel("Signal [V]")
        plt.title(f"Drive: {power.split('.')[0]} V")
        plt.legend()
        plt.close()

    voltages = [sorting_key(f) for f in files]
  
###Heights
for path in paths:
    plt.errorbar(voltages, heights[path.name], heights_err[path.name], [1e-3 for v in voltages],  fmt="+",
        capsize=3,
        markersize=6,
        color = "black",#colours[path.name],
        alpha = 0.5,
        linestyle="none",) #errors at some point
    popt, pcov = curve_fit(height_model, voltages, heights[path.name],sigma = heights_err[path.name], p0=[0.5, 1])
    y_fit = height_model(voltages, *popt)
    print(path.name, popt)
    plt.plot(voltages, y_fit, linestyle = "--", color = colours[path.name], label = labels[path.name],)


plt.ylabel(f"Height [V]")
plt.title(f"Peak Height vs Forcing")

plt.xlabel("Forcing [V]")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.show()

####Widths
mu_b = 9.274e-24 #J/T Bohr magneton
h = 6.626e-34 #J*s Planck's constant
conversion = {"85":1/3*(mu_b/h)*4.57e-7, "87": 1/2*(mu_b/h)*1.376e-7} #s to T
for path in paths:
    widths[path.name] = np.asarray(widths[path.name],  dtype=float)*conversion[path.name]
    widths_err[path.name] = np.asarray(widths_err[path.name],  dtype=float)*conversion[path.name]
    plt.errorbar(voltages, widths[path.name], widths_err[path.name] , [1e-3 for v in voltages], fmt="+",
        capsize=2,
        markersize=6,
        color = "black",#colours[path.name],
        alpha = 0.5,
        linestyle="none",)
#fit
    popt, pcov = curve_fit(widths_model, voltages, widths[path.name], p0=[0.5, 1])
    y_fit = widths_model(voltages, *popt)
    linewidth = float(f"{popt[0]:.3g}")
    linewidth_err_stat = float(f"{np.sqrt(pcov[0,0]):.2g}")
    linewidth_err_sys = float(f"{linewidth*0.043:.2g}")
    plt.plot(
        voltages,
        y_fit,
        linestyle="--",
        label=(
            rf"{labels[path.name]}: "
            rf"$\Gamma_0 = ({linewidth:g} \pm {linewidth_err_stat:g}_{{stat}}\pm{linewidth_err_sys:g}_{{sys}})\,\mathrm{{Hz}}$"
        ),
        color=colours[path.name]
    )    
    print(f"Natural linewidth {path.name}: ({popt[0]} \pm {np.sqrt(pcov[0][0])} Hz")

plt.title(f"Width vs Forcing")
plt.ylabel(f"Width [Hz]")
    
plt.xlabel("Forcing [V]")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()
#gains and times of sweep
#Big one
#For 25mV, 50mV, used max gain (1000), 3s time constant
