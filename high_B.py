import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from pathlib import Path
import scipy

path = Path("high_B/85")

plt.figure(figsize=(10, 6))

offset_step = 2.8   # vertical spacing
colors = ["black", "green"]
linestyles = ["--", ":"]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 25,
    "font.size": 16,
    "legend.fontsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 16,
})

# Track global x-range
all_x_min = 0
all_x_max = -np.inf
high_B = pd.read_csv("High_B.csv", skiprows=1).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()

labels = {"87":r"$^{87}\mathrm{Rb}$",
    "85":r"$^{85}\mathrm{Rb}$"}

for i, fname in enumerate(sorted(path.glob("*.csv"))):
    # Extract label, e.g. "6-00.csv" → "6.00 MHz"
    base = os.path.basename(fname)
    freq_label = base.replace(".csv", "").replace("-", ".") + " MHz"
    
    # Load data
    df = pd.read_csv(fname, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    t = df[:, 0]
    signal = df[:, 1]


    # Find position of the minimum
    idx_min = np.argmin(signal)
    t_min = t[idx_min]
    s_min = signal[idx_min]

    # Shift minimum to (x = 0, y = 0)
    t_shifted = t - t_min 
    if i > 1:
        t_shifted = t_shifted + 0.07*i
    
    if i > 3:
        t_shifted = t_shifted + 0.07*i
    
    s_shifted = signal - signal[2000]

    # Apply vertical offset between traces
    offset_signal = s_shifted + i * offset_step

    # Update global x max for plotting
    all_x_max = max(all_x_max, np.max(t_shifted))

    
    # Choose alternating colour
    color = colors[i % len(colors)]
    line_sty = linestyles[i % len(linestyles)]

    # Plot
    plt.plot(-t_shifted, offset_signal, color = color, linewidth = 1.0)
    # Add label next to each curve
    end_file = {"85":"4-40.csv", "87":"6-50.csv"}
    peak_times = [-1.87, -1.294, -0.846, -0.352, 0.123, 0.598, 1.064, 1.531, 2.024, 2.581]
    peak_labels = {"85":[r"$|{2,2}>\rightarrow|2,1>$"] * 10, "87":[]}
    if fname.name == end_file[path.name]:
        for l, t in enumerate(peak_times):
            idx = np.argmin(np.abs(-t_shifted - t))
            plt.text(
                t-0.1,              # this is the x coordinate you want
                offset_signal[idx] + 0.1*(-1)**l,        # slightly above the signal
                peak_labels[path.name][l],
                ha="center",
                va="bottom",
                fontsize=14
            )

    plt.text(
        -2.4,
        np.max(offset_signal),   
        freq_label, 
        va="center",
        fontsize=12,                    # <<< larger labels
        color=color,
        bbox=dict(facecolor='white',    # <<< box around label
                edgecolor=color, 
                boxstyle='round,pad=0.2',
                linewidth=1)
    )

plt.xlabel("Shifted Sweep time [s]")
plt.ylabel("Signal [a.u.]")
plt.title(f"High B-field: {labels[path.name]}")
plt.xlim(-3,3)
plt.tight_layout()
plt.show()
