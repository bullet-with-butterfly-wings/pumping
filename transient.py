import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os


plotting = False
path = Path("rabi3/oscillations_pretty")  # Change to desired path

import numpy as np
from scipy.optimize import curve_fit

# rabi_decay(t, A, tau, omega, phi, offset) must already be defined

def fit_with_score(x, y, p0):
    popt, pcov = curve_fit(rabi_decay, x, y, p0=p0, maxfev=10000)
    yhat = rabi_decay(x, *popt)
    n = len(y); k = len(popt)
    ssr = np.sum((y - yhat)**2)
    # AIC and small-sample correction (AICc)
    aic = n * np.log(ssr / n) + 2*k
    aicc = aic + (2*k*(k+1)) / max(n - k - 1, 1)
    return popt, pcov, ssr, aicc

def weighted_avg_params(datasets, p0):
    """
    datasets: iterable of (x, y), e.g. [(df1[:,0], df1[:,1]), (df2[:,0], df2[:,1]), ...]
    returns: (p_mean, weights, all_popt)
    """
    results = []
    for (x, y) in datasets:
        try:
            popt, pcov, ssr, aicc = fit_with_score(x, y, p0)
            results.append((popt, aicc))
        except Exception:
            # if a fit fails, skip it (weight = 0)
            continue

    if not results:
        raise RuntimeError("All fits failed.")

    # Convert AICc to Akaike weights
    aiccs = np.array([r[1] for r in results])
    dA = aiccs - np.min(aiccs)
    w = np.exp(-0.5 * dA)
    w /= np.sum(w)

    popts = np.array([r[0] for r in results])  # shape: (m, k)
    p_mean = np.average(popts, axis=0, weights=w)

    return p_mean, w, popts


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E
# Select only rows where time is between -0.1 and 3.7

scanning_range = []
is_detuning = path.name == "detuning"
if is_detuning:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x[0])*(-1 if "m" in x else 1))
else:
    scanning_range = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x))

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

    for wavefront in files:
        df = pd.read_csv(path/ parameter / wavefront, skiprows=1).to_numpy()
        mask = (df[:, 2] > 0) # only oscillations
        df = df[mask]
        df = df[:2000,:]
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
        if parameter == "12":
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
    #histograms of the frequencies sample, plot mean and std
    plt.hist(frequencies_sample, bins=10)
    plt.axvline(np.mean(frequencies_sample), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(frequencies_sample) + np.std(frequencies_sample), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(frequencies_sample) - np.std(frequencies_sample), color='g', linestyle='dashed', linewidth=1)
    plt.title(f"Histogram of Rabi Frequencies for parameter {parameter}")
    plt.xlabel("Rabi Frequency [kHz]")
    plt.ylabel("Counts")
    plt.show()
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
