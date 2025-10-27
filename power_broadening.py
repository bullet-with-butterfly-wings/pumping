import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
import os
from scipy.signal import argrelextrema


path = Path("power_broadening/90_new")  # Change to desired path

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

calibration_happening = False
peaks = []
for power in files:
    df = pd.read_csv(path / power, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    """
    if power.split(".")[0]+"-calibration.csv" in calibration_files:
        calibration_happening = True
        calibration = pd.read_csv(path / (power.split(".")[0]+"-calibration.csv"), skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
        df[:, 1] -= np.interp(df[:, 0], calibration[:, 0], calibration[:, 1])
    else:
        calibration_happening = False
    """
    mask = (df[:, 0] >= start_time)
    df = df[mask]
    if len(peaks) == 0:
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.2, distance=1000, prominence=0.2, width=200)
    """
    if (len(peaks) != 5 and not calibration_happening) or (len(peaks) != 4 and calibration_happening):
        plt.plot(df[:, 0], df[:, 1], label="Data")
        plt.vlines(df[peaks, 0], ymin=min(df[:, 1]), ymax=max(df[:, 1]), color='r', linestyle='--', label='Detected Peaks')
        plt.show()
        continue
    """
    offset = df[np.argmax(df[:, 1]), 1]
    window = 500
    plt.plot(df[:, 0], df[:, 1], label="Data")
    for i, peak in enumerate(peaks):
        if i == 2:
            continue
        try:
            popt, pcov = curve_fit(lambda x, height, width, pos: peak_profile(x, height, width, pos, offset), df[max(peak-window, 0):min(peak+window, df.shape[0])-1, 0], df[max(peak-window, 0):min(peak+window, df.shape[0])-1, 1], p0=[1, 0.2, df[min(peak, df.shape[0])-100, 0]])
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
        if peak == "0":
            continue
        plt.plot([sorting_key(f) for f in files], parameter[peak], label=f"{peak}", marker = "o")
    plt.title(f"{label[i]} vs Power - {path.name} kHz")
    plt.xlabel("Drive [V]")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(f"{label[i]} [V]")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.show()
    