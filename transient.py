import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit


plotting = False
path = Path("rabi2/oscillations")


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E
# Select only rows where time is between -0.1 and 3.7

files = [f.name for f in path.glob("*.csv")]  # Sort by file name
frequencies = []
lifetime = []
total = None

for wavefront in files:
    df = pd.read_csv(path / wavefront, skiprows=1).to_numpy()
    mask = (df[:, 2] > 0) # only oscillations
    df = df[mask]
    try:
        popt, pcov = curve_fit(rabi_decay, df[:, 0], df[:, 1], p0=[200, 1, 6.28, 0, 0])
    except (RuntimeError, ValueError, Warning) as e:
        print(f"Could not fit {wavefront}")
        continue

    frequencies.append(popt[2]/(2*np.pi))  # Convert angular frequency to frequency
    lifetime.append(1/popt[1])  # Store lifetime
    if total is None:
        total = df[:7000, 1]
    else:
        total += df[:7000, 1]
    if plotting:
        plt.plot(df[:, 0], df[:, 1], label=wavefront)
        plt.plot(df[:, 0], rabi_decay(df[:, 0], *popt), label=f"Fit {wavefront}")
        plt.xlabel("Time [ms]")
        plt.ylabel("Signal [V]")
        plt.legend()
        plt.show()

plt.plot(df[:7000, 0]*0.0064, total/len(files))
try:
    popt, pcov = curve_fit(rabi_decay, df[:7000, 0]*0.0064, total/len(files), p0=[200, 1, 6.28, 0, 0])
except (RuntimeError, ValueError, Warning) as e:
    print(f"Could not fit {wavefront}")

plt.plot(df[:7000, 0]*0.0064, rabi_decay(df[:7000, 0]*0.0064, *popt), label=f"Fit average")
plt.xlabel("Time [ms]")
plt.ylabel("Signal [V]")
plt.show()
print(f"Frequencies (kHz): {np.mean(frequencies)} +- {np.std(frequencies)/np.sqrt(len(frequencies))}")
print(f"Lifetime (ms): {np.mean(lifetime)} +- {np.std(lifetime)/np.sqrt(len(lifetime))}")

