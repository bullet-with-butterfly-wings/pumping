import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os


plotting = False
path = Path("rabi87")  # Change to desired path

import numpy as np
from scipy.optimize import curve_fit


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E
# Select only rows where time is between -0.1 and 3.7

scanning_range = []
is_detuning = path.name == "detuning"
if is_detuning:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x[0])*(-1 if "m" in x else 1))
else:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))
scanning_range = scanning_range
#scanning_range = scanning_range[:7]  # Use all voltages
frequencies = []
lifetimes = []
amplitude = []
lifetimes_error = []
frequencies_error = []
amplitude_error = []



for parameter in scanning_range:
    files = [f.name for f in path.glob(f"{parameter}/*.csv")]  # Sort by file name
    frequencies_sample = []
    lifetime_sample = [] #data from this voltage
    amplitude_sample = []

    chi2_distro = []
    for wavefront in files:
        df = pd.read_csv(path/ parameter / wavefront, skiprows=1).to_numpy()
        mask = (df[:, 2] > 0) # only oscillations
        df = df[mask]
        df = df[:2000,:]
        try:
            popt, pcov = curve_fit(rabi_decay, df[:, 0], df[:, 1], p0=[200, 1, 6.28, 0, 0])
            chi2 = np.sum(((df[:, 1] - rabi_decay(df[:, 0], *popt)) ** 2)/(abs(rabi_decay(df[:, 0], *popt)**2)))
            chi2_distro.append(chi2)
        except (RuntimeError, ValueError, Warning, Exception) as e:
            print(f"Could not fit {wavefront}")
            continue
    threshold = sorted(chi2_distro)[-1]  # 80th percentile
    for wavefront in files:
        df = pd.read_csv(path/ parameter / wavefront, skiprows=1).to_numpy()
        mask = (df[:, 2] > 0) # only oscillations
        df = df[mask]
        df = df[:2000,:]
        try:
            popt, pcov = curve_fit(rabi_decay, df[:, 0], df[:, 1], p0=[200, 1, 6.28, 0, 0])
            chi2 = np.sum(((df[:, 1] - rabi_decay(df[:, 0], *popt)) ** 2)/abs(rabi_decay(df[:, 0], *popt)))
            if chi2 > threshold:
                continue
            plt.plot(df[:, 0], df[:, 1], label=f"Data {wavefront}")
            plt.plot(df[:, 0], rabi_decay(df[:, 0], *popt), label=f"Fit {wavefront}")
            plt.legend()
            plt.close()
        except (RuntimeError, ValueError, Warning, Exception) as e:
            print(f"Could not fit {wavefront}")
            continue

        frequencies_sample.append(popt[2]/(2*np.pi))  # Convert angular frequency to frequency
        lifetime_sample.append(1/popt[1])  # Store lifetime
        amplitude_sample.append(popt[0]*np.exp(-popt[1]*df[0, 0]))  # Store amplitude
        """
        if total is None:
            total = df[:7000, 1]
        else:
            total += df[:7000, 1]
        """
    frequencies.append(np.mean(frequencies_sample))
    lifetimes.append(np.mean(lifetime_sample))
    frequencies_error.append(np.std(frequencies_sample))
    lifetimes_error.append(np.std(lifetime_sample))
    amplitude.append(np.mean(amplitude_sample))
    amplitude_error.append(np.std(amplitude_sample))
    #histograms of the frequencies sample, plot mean and std
    """
    plt.hist(frequencies_sample, bins=10)
    plt.axvline(np.mean(frequencies_sample), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(frequencies_sample) + np.std(frequencies_sample), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(frequencies_sample) - np.std(frequencies_sample), color='g', linestyle='dashed', linewidth=1)
    plt.title(f"Histogram of Rabi Frequencies for parameter {parameter}")
    plt.xlabel("Rabi Frequency [kHz]")
    plt.ylabel("Counts")
    """

buffer = []
if is_detuning:
    for v in scanning_range:
        if "m" in v:
            buffer.append(int(v[0])*-1)
        else:
            buffer.append(int(v[0]))
else:
    for v in scanning_range:
        if int(v) % 5 != 0:
            buffer.append(int(v)+0.5)
        else:
            buffer.append(int(v))
        
scanning_range = buffer
plt.plot(scanning_range, frequencies, marker="o")
plt.errorbar(scanning_range, frequencies, yerr=frequencies_error)
#fit straight line
if not is_detuning:
    coeffs = np.polyfit(scanning_range, frequencies, 1)
    fit_line = np.poly1d(coeffs)
    plt.plot(scanning_range, fit_line(scanning_range), linestyle='--', label=f"{coeffs[0]:.2e}*V + {coeffs[1]:.2e}")
xlabel = "Voltage [V]" if not is_detuning else "Detuning [kHz]"
plt.xlabel(xlabel)
plt.ylabel("Rabi Frequency [kHz]")
plt.title(f"Rabi Frequency vs {'Detuning' if is_detuning else 'Voltage'}")
plt.legend()
plt.show()

plt.plot(scanning_range, lifetimes, marker="o")
plt.errorbar(scanning_range, lifetimes, yerr=lifetimes_error)
plt.xlabel(xlabel)
plt.ylabel("Lifetime [ms]")
plt.title(f"Rabi Lifetime vs {'Detuning' if is_detuning else 'Voltage'}")
plt.show()

plt.plot(scanning_range, amplitude, marker="o")
plt.errorbar(scanning_range, amplitude, yerr=amplitude_error)
plt.xlabel(xlabel)
plt.ylabel("Amplitude [V]")
plt.title(f"Rabi Amplitude vs {'Detuning' if is_detuning else 'Voltage'}")
plt.show()

"""
plt.plot(df[:7000, 0]*0.0064, total/len(files))
try:
    popt, pcov = curve_fit(rabi_decay, df[:7000, 0]*0.0064, total/len(files), p0=[200, 1, 6.28, 0, 0])
except (RuntimeError, ValueError, Warning) as e:
    print(f"Could not fit {wavefront}")

plt.plot(df[:7000, 0]*0.0064, rabi_decay(df[:7000, 0]*0.0064, *popt), label=f"Fit average")
plt.xlabel("Time [ms]")
plt.ylabel("Signal [V]")
plt.show()
"""
