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
calibration_files = list(filter(lambda x: "calibration" in x, files))
files = list(filter(lambda x: "calibration" not in x, files))
start_time = 0.3 #s
print(files)
def peak_profile(x, height0, width0, pos0, offset): #Lorentzian profile
    return -height0 / (1 + ((x - pos0) / width0) ** 2) + offset

peaks_name = ["85-", "87-", "0", "87+", "85+"]
heights = {"85-": [], "87-": [], "0": [], "87+": [], "85+": []}
widths = {"85-": [], "87-": [], "0": [], "87+": [], "85+": []}
"""
peaks_name = ["Double photon"]
heights = {"Double photon": []}
widths = {"Double photon": []}
"""
peaks_name = ["Main"]
heights = {"Main": []}
widths = {"Main": []}

peaks = []
for power in files:
    df = pd.read_csv(path / power, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    mask = (df[:, 0] >= start_time)
    df = df[mask]
    """
    if power == "0.20.csv" or power == "0.10.csv":
        df[:, 0] = df[:, 0]/2.5
    if power == "0-40.csv":
        df[:, 1] = 2*df[:, 1]
        df[:, 0] = df[:, 0]
    if power == "0-80.csv" or power == "1-60.csv":
        df[:, 1] = 5*df[:, 1]
        df[:, 0] = df[:, 0]/2
    if sorting_key(power) > 1.7:
        df[:,1] = 10*df[:,1]
    """
    if len(peaks) == 0:
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.2, distance=1000, prominence=0.2, width=200)
    else:
        previous_peaks = peaks
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.2, distance=1000, prominence=0.2, width=200)
        if len(peaks) != len(previous_peaks):
            peaks = previous_peaks
    offset = df[np.argmax(df[:, 1]), 1]
    window = 2000
    plt.plot(df[:, 0], df[:, 1], label="Data")
    for i, peak in enumerate(peaks):
        if i == 2:
            continue
        try:
            popt, pcov = curve_fit(lambda x, height, width, pos, off: peak_profile(x, height, width, pos, off), df[max(peak-window, 0):min(peak+window, df.shape[0])-1, 0], df[max(peak-window, 0):min(peak+window, df.shape[0])-1, 1], p0=[1, 0.2, df[min(peak, df.shape[0])-1, 0], offset])
            plt.plot(df[:, 0], peak_profile(df[:, 0], *popt), label=f"Fit {peaks_name[i]}")     
        except (RuntimeError, ValueError) as e:
            plt.close()
            plt.vlines(df[peak-window, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peaks_name[i]}")
            plt.vlines(df[min(peak+window, df.shape[0])-1, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')
            plt.plot(df[:, 0], df[:, 1], label="Data")
            plt.show()
            print(f"Could not fit {power}")
            continue
        heights[peaks_name[i]].append(abs(popt[0]))
        widths[peaks_name[i]].append(abs(popt[1]))
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [V]")
    plt.title(f"Drive: {power.split('.')[0]} V")
    plt.legend()
    plt.show()

label = ["Heights", "Widths"]
for i, parameter in enumerate([heights, widths]):
    for peak in peaks_name:
        if peak == "0":
            continue
        plt.plot([sorting_key(f) for f in files], parameter[peak], label=f"{peak}", marker = "o")
    plt.title(f"{label[i]} vs Power - {path.name} kHz")
    if parameter == heights: #fit power law
        def power_law(x, a, b):
            return a * x ** b
        popt, pcov = curve_fit(power_law, [sorting_key(f) for f in files], parameter[peak], p0=[1, 4])
        print(f"Fitted parameters for {label[i]}: a = {popt[0]}, b = {popt[1]}")
        x_fit = np.linspace(min([sorting_key(f) for f in files]), max([sorting_key(f) for f in files]), 100)
        plt.plot(x_fit, power_law(x_fit, *popt), label=f"Fit: {popt[0]:.2e} * x^{popt[1]:.2f}", color='black', linestyle='--')
    plt.xlabel("Drive [V]")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(f"{label[i]} [V]")
    plt.legend()
    plt.show()
    

#gains and times of sweep
#Big one
#For 25mV, 50mV, used max gain (1000), 3s time constant
gains = [20, 20, 20, 20, 20, 20, 20, 20, 50, 100,  500, 1000, 1000]