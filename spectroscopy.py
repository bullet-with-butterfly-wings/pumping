import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("e2.csv")

# Replace missing values with -1
df = df.replace(r'^\s*$', np.nan, regex=True)  # convert empty strings to NaN
df = df.fillna(-1)

# Try to convert all numeric columns, leave text columns as-is
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        pass  # skip columns that contain text

frequencies = df["f"].values
base = df["base"].values
conversion = 0.6/10 #s to Gauss
conversion = 0.058
time_of_sweep = 10.4 #0.05 = 0.5%
mu_0 = 1.2566e-2 #in Gauss
V = 0.995 #0.2%
R = 1.0 #1% 
B_max = (4/5)**(3/2)*11*(V/R)*mu_0/(0.1639) 
print(B_max)
conversion = B_max/time_of_sweep #all together ignore Hemholtz condition (good stability) and Radii uncertainty = 1.5%
print(conversion)

#constants
mu_b = 9.274e-24 #J/T Bohr magneton
h = 6.626e-34 #J*s Planck's constant


peaks = ["small_plus", "small_minus", "big_plus", "big_minus"]

peaks_label = [
    r"$^{87}\mathrm{Rb}$",
    r"$^{85}\mathrm{Rb}$",
]

peaks_colour = ["r", "g"]
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
for i, peak in enumerate(peaks):
    peak_values = df[peak].values
    valid_freqs = [f for f in frequencies if peak_values[list(frequencies).index(f)] != -1] #frequencies
    peak_values = [(v-base[list(peak_values).index(v)])*conversion for v in peak_values if v != -1] #B field in G

    coeffs, cov = np.polyfit(peak_values, valid_freqs, 1, cov = True)
    fit_line = np.poly1d(coeffs)
    plt.plot([0]+peak_values, fit_line([0]+peak_values), linestyle='--', color = peaks_colour[(i)//2], label = (peaks_label[i//2] if i % 2 == 0 else None))

    # 1.5% error bars on B field
    peak_err = [0.015 * p for p in peak_values]

    # plot data with error bars
    plt.errorbar(
        peak_values,
        valid_freqs,
        xerr=peak_err,
        fmt="+",
        capsize=2,
        markersize=4,
        color = "black",
        alpha = 0.5,
        linestyle="none",
    )
    print(peak)
    print("a", coeffs[0])
    print("b", coeffs[1])
    print("delta a", np.sqrt(cov[0,0]))
    print("delta b", np.sqrt(cov[1,1]))
    print(f"g_F = {(coeffs[0]*h*1e3*1e4)/(mu_b)} +- {(coeffs[0]*h*1e3*1e4)/(mu_b)*0.015}")


#coeffs[0] = f/B
#f"$f = {coeffs[0]:.1e}({np.sqrt(cov[0][0]):.1e})B + {coeffs[1]:.1e}({np.sqrt(cov[1][1]):.1e})$"
plt.text(0.05, 180, r"$\mathbf{g_F^{(87+)} = +0.494(7)}$", color="black", fontsize=12)
plt.text(0.18, 60, r"$\mathbf{g_F^{(85+)} = +0.329(5)}$", color="black", fontsize=12)
plt.text(-0.28, 40, r"$\mathbf{g_F^{(85-)} = -0.330(5)}$", color="black", fontsize=12)
plt.text(-0.20, 160, r"$\mathbf{g_F^{(87-)} = -0.495(7)}$", color="black", fontsize=12)


plt.grid()
plt.legend()
plt.title(f"Zeeman Splitting vs Magnetic Field")
plt.ylabel("Frequency [kHz]")
plt.xlabel("B field [G]")
plt.tight_layout()
plt.show()


#read the high_B.csv file and plot the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Load and clean data ---
df = pd.read_csv("power_broadening/100/20.csv", skiprows=2)
df = df.apply(pd.to_numeric, errors='coerce').dropna().to_numpy()

t = df[5000:, 0] * conversion
signal = df[5000:, 1]

# --- Find dips (negative peaks) ---
peaks, _ = find_peaks(-signal, distance=1000, prominence=0.05)

# --- Plot ---

plt.figure(figsize=(6, 4))
plt.plot(t, signal, color="orange", lw=1.2)
plt.xlabel(r"$\Delta B\ \mathrm{[G]}$", fontsize=16)
plt.ylabel(r"Signal\ [V]", fontsize=12)
plt.ylim(bottom = -1.2)
plt.title(r"\textbf{Spectroscopy}", fontsize=16)
plt.tick_params(direction='in', top=True, right=True)

# --- Peak labels ---
labels = [
    r"$^{85}\mathrm{Rb}(-)$",
    r"$^{87}\mathrm{Rb}(-)$",
    r"$B = 0$",
    r"$^{87}\mathrm{Rb}(+)$",
    r"$^{85}\mathrm{Rb}(+)$",

]
# Choose four most prominent peaks (sorted left to right)
selected_peaks = sorted(peaks[np.argsort(t[peaks])[:5]])
for i, p in enumerate(selected_peaks):
    plt.text(
        t[p],
        signal[p] - 0.10,
        labels[i],
        ha='center',
        va='bottom',
        fontsize=16,
        color='black'
    )

plt.tight_layout()
plt.show()
