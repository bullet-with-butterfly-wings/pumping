import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os

paths = [Path("rabi85"), Path("rabi87")]
labels = {
    "rabi87": r"$^{87}\mathrm{Rb}$",
    "rabi85": r"$^{85}\mathrm{Rb}$",
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


def rabi_decay(t, A, B, C, D, E):
    return A * np.exp(-B * t) * np.cos(C * t + D) + E

def W_fit(V, delta, A):
    return np.sqrt(delta**2 + (A * V)**2)

def amplitude_fit(V, delta, A, k):
    return k * (A * V)**2 / ((A * V)**2 + delta**2)

def lifetime_fit(V, A, gamma):
    # your current model
    return 1 / np.sqrt(2 * (A * V)**2 + gamma**2)

def robust_mean(values, sigma_clip=2.0):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.nan
    m = np.mean(values)
    s = np.std(values)
    if s == 0:
        return m
    mask = np.abs(values - m) < sigma_clip * s
    if mask.sum() == 0:
        return m
    return np.mean(values[mask])

# store everything here

results = {}

for path in paths:
    # subfolders = voltages
    scanning_range = sorted(
        [x for x in os.listdir(path) if "." not in x],
        key=lambda x: int(x)
    )

    frequencies = []
    lifetimes = []
    amplitude = []
    frequencies_error = []
    lifetimes_error = []
    amplitude_error = []

    for parameter in scanning_range:
        files = [f.name for f in (path / parameter).glob("*.csv")]

        frequencies_sample = []
        lifetime_sample = []
        amplitude_sample = []

        for wavefront in files:
            df = pd.read_csv(path / parameter / wavefront, skiprows=1).to_numpy()

            # keep only oscillating region
            mask = df[:, 2] > 0
            df = df[mask][:4000, :]

            try:
                popt, pcov = curve_fit(
                    rabi_decay,
                    df[:, 0],
                    df[:, 1],
                    p0=[200, 1, 2*np.pi*1, 0, 0],
                    maxfev=5000
                )
            except Exception:
                print(f"Could not fit {path}/{parameter}/{wavefront}")
                continue

            A_r, B_r, C_r, D_r, E_r = popt
            frequencies_sample.append(C_r / (2*np.pi))
            lifetime_sample.append( B_r)
            if path.name == "rabi85":
                amplitude_sample.append(5*abs(A_r))
            else:
                amplitude_sample.append(abs(A_r))
            if path.name == "rabi85" and parameter == "5":
                plt.plot(df[:,0], df[:,1], label = "orig")
                plt.plot(df[:,0], rabi_decay(df[:,0], *popt))
                plt.close()

        # robust mean per voltage
        frequencies.append(robust_mean(frequencies_sample))
        lifetimes.append(robust_mean(lifetime_sample))
        amplitude.append(robust_mean(amplitude_sample))
        n = len(frequencies_sample)
        if n > 1:
            frequencies_error.append(np.std(frequencies_sample, ddof=1) / np.sqrt(n))
            lifetimes_error.append(np.std(lifetime_sample, ddof=1) / np.sqrt(n))
            amplitude_error.append(np.std(amplitude_sample, ddof=1) / np.sqrt(n))
        else:
            frequencies_error.append(0.0)
            lifetimes_error.append(0.0)
            amplitude_error.append(0.0)

    V = np.array([float(v) for v in scanning_range])

    # fit each quantity here and store
    res = {
        "V": V,
        "freq": np.array(frequencies),
        "freq_err": np.array(frequencies_error),
        "life": np.array(lifetimes),
        "life_err": np.array(lifetimes_error),
        "amp": np.array(amplitude),
        "amp_err": np.array(amplitude_error),
    }

    # lifetime fit
    try:
        p_life, _ = curve_fit(lifetime_fit, V, res["life"], p0=[0.2, res["life"].min()])
        res["life_fit_params"] = p_life
    except Exception:
        res["life_fit_params"] = None

    # amplitude fit
    try:
        p_amp, _ = curve_fit(amplitude_fit, V, res["amp"], p0=[1e3, 0.2, 1], bounds=(0, np.inf))
        res["amp_fit_params"] = p_amp
    except Exception:
        res["amp_fit_params"] = None

    # frequency fit
    try:
        p_freq, _ = curve_fit(W_fit, V, res["freq"], p0=[0.2, res["freq"].max()])
        res["freq_fit_params"] = p_freq
    except Exception:
        res["freq_fit_params"] = None

    results[path.name] = res

# =========================
# Now plot ALL on the same plots
# =========================
V_dense = np.linspace(
    0,
    max(r["V"].max() for r in results.values()),
    200
)

# 1) Lifetime
plt.figure()
colous = ["g", "r"]
for name, r in results.items():
    plt.errorbar(r["V"], r["life"], yerr=r["life_err"], fmt="+",
        capsize=2,
        markersize=4,
        color = ("r" if name == "rabi87" else "g"),
        alpha = 0.5,
        linestyle="none")
    if r["life_fit_params"] is not None:
        plt.plot(V_dense, lifetime_fit(V_dense, *r["life_fit_params"]), label=labels[name], linestyle = "--", color = ("r" if name == "rabi87" else "g"))
plt.xlabel("Voltage [V]")
plt.ylabel("Lifetime [s]")
plt.ylim(bottom = 0)
plt.title("Rabi lifetime vs Voltage")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Amplitudeplt.figure()
for name, r in results.items():
    plt.errorbar(r["V"], r["amp"], yerr=r["amp_err"],   fmt="+",
        capsize=2,
        markersize=4,
        color = ("r" if name == "rabi87" else "g"),
        alpha = 0.5,
        linestyle="none")
    if r["amp_fit_params"] is not None:
        plt.plot(V_dense, amplitude_fit(V_dense, *r["amp_fit_params"]), label=labels[name], linestyle = "--", color = ("r" if name == "rabi87" else "g"))
plt.xlabel("Voltage [V]")
plt.ylabel("Amplitude [arb.]")
plt.ylim(bottom = 0)
plt.title("Rabi Amplitude vs Voltage")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Frequency
plt.figure()
for name, r in results.items():
    plt.errorbar(r["V"], r["freq"], yerr=r["freq_err"],  fmt="+",
        capsize=2,
        markersize=4,
        color = "black",#("r" if name == "rabi87" else "g"),
        alpha = 0.6,
        linestyle="none")
    if r["freq_fit_params"] is not None:
        print(*r["freq_fit_params"])
        plt.plot(V_dense, W_fit(V_dense, *r["freq_fit_params"]), linestyle = "--", label=labels[name], color = ("r" if name == "rabi87" else "g"))
plt.xlabel("Voltage [V]")
plt.ylabel(r"$\sqrt{\Omega^2+\Delta^2}$ [kHz]")
plt.title("Oscillation frequency vs Voltage")
plt.legend()
plt.ylim(bottom = 0)
plt.tight_layout()
plt.show()


"""
#detuning
path = Path("rabi3/detuning")
folders = sorted(list(filter(lambda x: "." not in x, os.listdir(path))), key=lambda x: int(x[0])*(-1 if "m" in x else 1))
scanning_range = []
for v in folders:
    if "m" in v:
        scanning_range.append(int(v[0])*-1)
    else:
        scanning_range.append(int(v[0]))

amp = []
freq = []
amp_err = []
freq_err = []
for data_point in folders:
    freq_sample = []
    amp_sample = []
    files = [f.name for f in (path / data_point).glob("*.csv")]
    for wavefront in files:
        df = pd.read_csv(path / data_point/ wavefront, skiprows=1).to_numpy()
        # keep only oscillating region
        mask = df[:, 2] > 0
        df = df[mask][:4000, :]

        try:
            popt, pcov = curve_fit(
                rabi_decay,
                df[:, 0],
                df[:, 1],
                p0=[200, 1, 2*np.pi*1, 0, 0],
                maxfev=5000
            )
        except Exception:
            print(f"Could not fit {path}/{wavefront}")
            continue

        A_r, B_r, C_r, D_r, E_r = popt
        freq_sample.append(C_r / (2*np.pi))
        amp_sample.append(abs(A_r))
    freq.append(np.mean(freq_sample))
    freq_err.append(np.std(freq_sample)/np.sqrt(len(freq_sample)))
    amp.append(np.mean(amp_sample))
    amp_err.append(np.std(amp_sample)/np.sqrt(len(amp_sample)))


# --- Frequency vs Detuning ---

# Convert scanning_range to a NumPy array
x = np.array(scanning_range, dtype=float)

# --- Frequency vs Detuning ---
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# --- Define models ---
def model_no_offset(x, Omega, offset):
    return np.sqrt((x - offset)**2 + Omega**2)

def model_with_offset(x, Omega, offset, f_off):
    return np.sqrt((x - offset)**2 + Omega**2) + f_off

# --- Fit 1: No vertical offset ---
popt1, pcov1 = curve_fit(
    model_no_offset,
    x,
    freq,
    sigma=freq_err,
    absolute_sigma=True,
    p0=[500, 1],
    maxfev=10000
)
Omega1, offset1 = popt1
Omega1_err, offset1_err = np.sqrt(np.diag(pcov1))

# Compute chi-square
res1 = (freq - model_no_offset(x, *popt1)) / freq_err
chi2_1 = np.sum(res1**2)
ndof1 = len(x) - len(popt1)
redchi2_1 = chi2_1 / ndof1

# --- Fit 2: With vertical offset ---
popt2, pcov2 = curve_fit(
    model_with_offset,
    x,
    freq,
    sigma=freq_err,
    absolute_sigma=True,
    p0=[500, 1, 0],
    maxfev=10000
)
Omega2, offset2, f_off = popt2
Omega2_err, offset2_err, f_off_err = np.sqrt(np.diag(pcov2))

res2 = (freq - model_with_offset(x, *popt2)) / freq_err
chi2_2 = np.sum(res2**2)
ndof2 = len(x) - len(popt2)
redchi2_2 = chi2_2 / ndof2

# --- Likelihood ratio ---
delta_chi2 = chi2_1 - chi2_2
likelihood_ratio = np.exp(-0.5 * delta_chi2)

# --- Print summary ---
print("=== Chi-square results ===")
print(f"No-offset fit:     chi2 = {chi2_1:.2f},  dof = {ndof1},  chi2/dof = {redchi2_1:.2f}")
print(f"With-offset fit:   chi2 = {chi2_2:.2f},  dof = {ndof2},  chi2/dof = {redchi2_2:.2f}")
print(f"Dchi2 = {delta_chi2:.2f}")
print(f"Likelihood ratio (offset/non-offset) = {likelihood_ratio:.3g}")
print(f"Vertical offset f_off = {f_off:.3f} \pm {f_off_err:.3f} kHz")


# --- Plot ---
x_fit = np.linspace(x.min(), x.max(), 400)
plt.figure(figsize=(6,4))
plt.errorbar(x - offset1, freq, yerr=freq_err, fmt='+', capsize=3, color="black", alpha=0.6)
plt.plot(x_fit - offset1, model_no_offset(x_fit, *popt1), '--', color="blue",
         label=fr"No offset: $\Omega={Omega1:.1f}\pm{0.1}$ kHz")
plt.plot(x_fit - offset2, model_with_offset(x_fit, *popt2), ':', color="red",
         label=fr"With offset: $\Omega={Omega2:.1f}\pm{Omega2_err:.1f}$ kHz")
plt.xlabel(r"Detuning  $\Delta$ [kHz]")
plt.ylabel(r"$\sqrt{\Omega^2+\Delta^2}$ [kHz]")
plt.title("Frequency vs Detuning")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --- Amplitude vs Detuning ---
def amp_model(x, Omega, A0, offset):
    return A0 * (Omega**2) / (Omega**2 + (x-offset)**2)

popt, pcov = curve_fit(
    amp_model,
    x,
    amp,
    sigma=amp_err,
    absolute_sigma=True,
    p0=[5, max(amp), 1],
    maxfev=10000
)
Omega_a, A0, offset = popt

Omega_a_err, A0_err, offset_err = np.sqrt(np.diag(pcov))
print(offset)
plt.figure(figsize=(6,4))
plt.errorbar(x - offset, amp, yerr=amp_err, fmt='+', capsize=3, color = "black", alpha = 0.6)
plt.plot(x_fit - offset, amp_model(x_fit, Omega_a, A0, offset), '--', color = "blue",
         label=(fr"Fit theory: $\Omega={Omega_a:.1f}\pm{Omega_a_err:.1f}$ kHz"))
plt.xlabel(r"Detuning $\Delta$ [kHz]")
plt.ylabel("Amplitude [a.u.]")
plt.title("Amplitude vs Detuning")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
"""