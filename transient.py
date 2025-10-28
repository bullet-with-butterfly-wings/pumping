import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os


paths = [Path("rabi87"), Path("rabi85")]


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E
# Select only rows where time is between -0.1 and 3.7

scanning_range = []
"""
is_detuning = path.name == "detuning"
if is_detuning:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x[0])*(-1 if "m" in x else 1))
else:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))
scanning_range = scanning_range
#scanning_range = scanning_range[:7]  # Use all voltages
"""
f = {}
f_err = {}
for path in paths:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))
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
            """
            if path.name == "rabi85":
                df[:,1] = df[:,1]/20
            if path.name == "rabi87":
                df[:,1] = df[:,1]/100
            """
            try:
                popt, pcov = curve_fit(rabi_decay, df[:, 0], df[:, 1], p0=[200, 1, 6.28, 0, 0])
            except (RuntimeError, ValueError, Warning, Exception) as e:
                print(f"Could not fit {wavefront}")
                continue
    
            frequencies_sample.append(popt[2]/(2*np.pi))  # Convert angular frequency to frequency
            lifetime_sample.append(1/popt[1])  # Store lifetime
            amplitude_sample.append(popt[0]*np.exp(-popt[1]*df[0, 0]))  # Store amplitude
        frequencies.append(np.mean(frequencies_sample))
        lifetimes.append(np.mean(lifetime_sample))
        frequencies_error.append(np.std(frequencies_sample))
        lifetimes_error.append(np.std(lifetime_sample))
        amplitude.append(np.mean(amplitude_sample))
        amplitude_error.append(np.std(amplitude_sample))
        f[path.name] = frequencies
        f_err[path.name] = frequencies_error
    plt.plot(scanning_range, lifetimes, marker="o")
    plt.errorbar(scanning_range, lifetimes, yerr=lifetimes_error)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Lifetime [ms]")
    plt.title(f"Rabi Lifetime vs Voltage {path.name}")
    plt.show()

    plt.plot(scanning_range, amplitude, marker="o")
    plt.errorbar(scanning_range, amplitude, yerr=amplitude_error)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Amplitude [V]")
    plt.title(f"Rabi Amplitude vs Voltage {path.name}")
    plt.show()
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
"""     


for path in paths:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))
    buffer = []
    for v in scanning_range:
            if int(v) % 5 != 0:
                buffer.append(int(v)+0.5)
            else:
                buffer.append(int(v))
    scanning_range = buffer
    print(scanning_range)
    plt.plot(scanning_range, f[path.name], marker="o")
    plt.errorbar(scanning_range, f[path.name], yerr=f_err[path.name], fmt='o')
    #fit straight line
    coeffs, cov = np.polyfit(scanning_range, f[path.name], 1, cov =True)
    fit_line = np.poly1d(coeffs)
    plt.plot(scanning_range, fit_line(scanning_range), linestyle='--', label=f"{path.name}: ({coeffs[0]:.2e} +- {np.sqrt(cov[0, 0]):.0e})*V + {coeffs[1]:.2e} +- {np.sqrt(cov[1, 1]):.0e}")
plt.xlabel("Voltage [V]")
plt.ylabel("Rabi Frequency [kHz]")
plt.title(f"Rabi Frequency vs Voltage")
plt.legend()
plt.show()

"""
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
