import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
import os
from scipy.signal import argrelextrema


path = Path("power_broadening/big")  # Change to desired path

def sorting_key(file_name):
    name = file_name.split(".")[0].split("-")
    if len(name) > 1:
        return int(name[0]) + int(name[1])/100
    else:
        return int(name[0])

files = sorted([f.name for f in path.glob("*.csv")], key=sorting_key, reverse=True)  # Sort by file name
start_time = 3 #s
print(files)

def peak_profile(x, height0, width0, pos0, offset, drift): #Lorentzian profile
    return -height0 / (1 + ((x - pos0) / width0) ** 2) + offset + drift*x

def power_law(x, a, b):
    return a * x**b 
peaks = []
gains = {"big": [20, 20, 20, 20, 20, 20, 20, 20, 50, 100, 500, 1000_000, 1000_000]}
gains = {"big": [1]*13}
#gains = {"big": [20, 20, 20, 20, 20, 20, 20, 20, 50, 100, 500, 1000, 1000]}
heights = []
widths = []
for power in files:
    df = pd.read_csv(path / power, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    mask = (df[:, 0] >= start_time)
    df = df[mask]
    df[:, 1] = df[:, 1]/gains[path.name][files.index(power)]
    
    if len(peaks) == 0:
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.01, distance=100, prominence=0.02, width=200)
    else:
        previous_peaks = peaks
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.01, distance=1000, prominence=0.2, width=200)
        if len(peaks) != len(previous_peaks):
            peaks = previous_peaks
    plt.plot(df[:, 0], df[:, 1], label="Data")
    for i, peak in enumerate(peaks):
        if i == 2:
            continue
        try:
            if False:
                window = 10_000
                df = df[peak-window:peak+window,:]
                peak = window
            popt, pcov = curve_fit(lambda x, height, width, pos, off, drift: peak_profile(x, height, width, pos, off, drift), df[:, 0], df[:, 1], p0=[df[peak, 1], df[peak, 1]/3, df[peak, 0], 0, 0])
            previous_peaks = popt[2]
            plt.plot(df[:, 0], peak_profile(df[:, 0], *popt), label=f"Fit {peak}")

        except (RuntimeError, ValueError) as e:
            plt.close()
            plt.vlines(df[peak, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peak}")
            plt.vlines(df[min(peak, df.shape[0]-1), 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')
            plt.plot(df[:, 0], df[:, 1], label="Data")
            plt.show()
            print(f"Could not fit {power}: {e}")
            continue
        heights.append(abs(popt[0]))
        widths.append(abs(popt[1]))

    plt.xlabel("Time [s]")
    plt.ylabel("Signal [V]")
    plt.title(f"Drive: {power.split('.')[0]} V")
    plt.legend()
    plt.close()

voltages = [sorting_key(f) for f in files]

def heights_model(x, a, b):
    x = np.asarray(x, dtype=float)
    return a*(x**2)/(1 + b*(x**2))

popt, pcov = curve_fit(heights_model, voltages, heights, p0=[1, 0.5])
y_fit = heights_model(voltages, *popt)
plt.plot(voltages, y_fit, label="Heights Model Fit", linestyle="--")
plt.plot(voltages, heights, label="Heights", marker="o")
plt.xlabel("Drive [V]")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(f"Heights [V]")
plt.legend()
plt.show()

def widths_model(x, gamma, a):
    x = np.asarray(x, dtype=float)
    return gamma/2*np.sqrt(1+ 2*abs(a)*(x**2))
popt, pcov = curve_fit(widths_model, voltages, widths, p0=[0.5, 1])
y_fit = widths_model(voltages, *popt)
print("Natural linewidth:", popt[0], "s of sweep time")

plt.plot(voltages, y_fit, label=f"Fit", linestyle="--")
plt.plot(voltages, widths, label="Widths", marker="o")
plt.xlabel("Drive [V]")
plt.ylabel(f"Widths [s]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

#gains and times of sweep
#Big one
#For 25mV, 50mV, used max gain (1000), 3s time constant
