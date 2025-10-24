import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os


plotting = False
path = Path("rabi3/oscillations_pretty")


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E
# Select only rows where time is between -0.1 and 3.7

voltage_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))
frequencies = []
lifetimes = []
amplitude = []
lifetimes_error = []
frequencies_error = []
amplitude_error = []


for voltage in voltage_range:
    files = [f.name for f in path.glob(f"{voltage}/*.csv")]  # Sort by file name
    frequencies_sample = []
    lifetime_sample = [] #data from this voltage
    amplitude_sample = []
    total = None

    for wavefront in files:
        df = pd.read_csv(path/ voltage / wavefront, skiprows=1).to_numpy()
        mask = (df[:, 2] > 0) # only oscillations
        df = df[mask]
        #df = df[:1000,:]
        try:
            popt, pcov = curve_fit(rabi_decay, df[:, 0], df[:, 1], p0=[200, 1, 6.28, 0, 0])
        except (RuntimeError, ValueError, Warning) as e:
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
        if plotting:
            plt.plot(df[:, 0], df[:, 1], label=wavefront)
            plt.plot(df[:, 0], rabi_decay(df[:, 0], *popt), label=f"Fit {wavefront}")
            plt.xlabel("Time [ms]")
            plt.ylabel("Signal [V]")
            plt.legend()
            plt.show()
    frequencies.append(np.mean(frequencies_sample))
    lifetimes.append(np.mean(lifetime_sample))
    frequencies_error.append(np.std(frequencies_sample))
    lifetimes_error.append(np.std(lifetime_sample))
    amplitude.append(np.mean(amplitude_sample))
    amplitude_error.append(np.std(amplitude_sample))

buffer = []
for v in voltage_range:
    if int(v) % 5 != 0:
        buffer.append(int(v)+0.5)
    else:
        buffer.append(int(v))
voltage_range = buffer 
plt.plot(voltage_range, frequencies, marker="o")
plt.errorbar(voltage_range, frequencies, yerr=frequencies_error)
#fit straight line
coeffs = np.polyfit(voltage_range, frequencies, 1)
fit_line = np.poly1d(coeffs)
plt.plot(voltage_range, fit_line(voltage_range), linestyle='--', label=f"{coeffs[0]:.2e}*V + {coeffs[1]:.2e}")
plt.xlabel("Voltage [V]")
plt.ylabel("Rabi Frequency [kHz]")
plt.title("Rabi Frequency vs Voltage")
plt.legend()
plt.show()

plt.plot(voltage_range, lifetimes, marker="o")
plt.errorbar(voltage_range, lifetimes, yerr=lifetimes_error)
plt.xlabel("Voltage [V]")
plt.ylabel("Lifetime [ms]")
plt.title("Rabi Lifetime vs Voltage")
plt.show()

plt.plot(voltage_range, amplitude, marker="o")
plt.errorbar(voltage_range, amplitude, yerr=amplitude_error)
plt.xlabel("Voltage [V]")
plt.ylabel("Amplitude [V]")
plt.title("Rabi Amplitude vs Voltage")
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
