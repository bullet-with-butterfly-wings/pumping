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

print(df)

frequencies = df["f"].values
base = df["base"].values
conversion = 0.6/10 #s to Gauss

#constants
mu_b = 9.274e-24 #J/T Bohr magneton
h = 6.626e-34 #J*s Planck's constant

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

peaks = ["small_plus", "small_minus", "big_plus", "big_minus"]
for i, peak in enumerate(peaks):
    peak_values = df[peak].values
    valid_freqs = [f for f in frequencies if peak_values[list(frequencies).index(f)] != -1]
    peak_values = [(v-base[list(peak_values).index(v)])*conversion for v in peak_values if v != -1]
    axes[i].plot(valid_freqs, peak_values, marker = "o",markersize=4, label=peak)
    #fit straight line
    coeffs = np.polyfit(valid_freqs[3:], peak_values[3:], 1)
    fit_line = np.poly1d(coeffs)
    print(coeffs[0])
    axes[i].plot(valid_freqs, fit_line(valid_freqs), linestyle='--', label=f"{coeffs[0]/1e3:.2e}*f + {coeffs[1]}")
    axes[i].grid()
    #print(f"{peak} fit: B = {coeffs[0]/1e3:.4f}f + {coeffs[1]:.4f}")
    print(f"{peak} g_F*m_F =", (h*1e3*1e4)/(coeffs[0]*mu_b))
    axes[i].legend()
    axes[i].set_title(f"{peak}     g_F*m_F = {(h*1e3*1e4)/(coeffs[0]*mu_b):.3f}")
    axes[i].set_xlabel("Frequency [kHz]")
    axes[i].set_ylabel("B field [G from the broad peak]")

plt.suptitle("Frequency vs B field for different peaks")
plt.tight_layout()
plt.show()
