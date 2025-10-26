import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
import os
from scipy.signal import argrelextrema


path = Path("power_broadening/100")  # Change to desired path

files = sorted([f.name for f in path.glob("*.csv")], key=lambda x: int(x.split(".")[0]))  # Sort by file name
start_time = 0.3 #s

def peak_profile(x, height0, width0, pos0, offset): #Lorentzian profile
    return -height0 / (1 + ((x - pos0) / width0) ** 2) + offset

peaks_name = ["85-", "87-", "0", "87+", "85+"]
heights = {"85-": [], "87-": [], "0": [], "87+": [], "85+": []}
widths = {"85-": [], "87-": [], "0": [], "87+": [], "85+": []}

for power in files:
    fitting = True
    df = pd.read_csv(path / power, skiprows=2).to_numpy()
    mask = (df[:, 0] >= start_time)
    df = df[mask]
    peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.05, distance=500, prominence=0.1, width = 20)
    offset = df[np.argmax(df[:, 1]), 1]
    window = 2000
    plt.plot(df[:, 0], df[:, 1], label="Data")
    for i, peak in enumerate(peaks):
        try:
            popt, pcov = curve_fit(lambda x, height, width, pos: peak_profile(x, height, width, pos, offset), df[peak-window:peak+window, 0], df[peak-window:peak+window, 1], p0=[1, 0.2, df[peak, 0]])
            plt.plot(df[:, 0], peak_profile(df[:, 0], *popt, offset), label=f"Fit {peaks_name[i]}")     
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit {power}")
            continue
        heights[peaks_name[i]].append(abs(popt[0]))
        widths[peaks_name[i]].append(abs(popt[1]))
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [V]")
    plt.title(f"Power: {power.split('.')[0]} V")
    plt.legend()
    plt.close()
label = ["Heights", "Widths"]
for i, parameter in enumerate([heights, widths]):
    for peak in peaks_name:
        plt.plot([int(f.split(".")[0]) for f in files], parameter[peak], label=f"{peak}")
    plt.xlabel("Power [V]")
    plt.ylabel(f"{label[i]} [V]")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.show()
    