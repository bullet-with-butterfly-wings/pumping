import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

path = "high_B/87/*.csv"

plt.figure(figsize=(10, 6))

offset_step = 2.5   # vertical spacing
colors = ["C0", "C1"]

# Track global x-range
all_x_min = 0
all_x_max = -np.inf

for i, fname in enumerate(sorted(glob.glob(path))):
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
    s_shifted = signal - signal[2000]

    # Apply vertical offset between traces
    offset_signal = s_shifted + i * offset_step

    # Update global x max for plotting
    all_x_max = max(all_x_max, np.max(t_shifted))

    # Choose alternating colour
    color = colors[i % len(colors)]

    # Plot
    plt.plot(t_shifted, offset_signal, color=color)

    # Add label next to each curve
    plt.text(
        2,
        np.max(offset_signal),
        freq_label,
        va="center",
        fontsize=8,
        color=color
    )

plt.xlabel("Time shifted so that minimum is at $t = 0$ [s]")
plt.ylabel("Signal + offset [a.u.]")
plt.title("High B-field: traces aligned so minima occur at $t=0$ (87Rb)")
plt.xlim(-3,3)
plt.tight_layout()
plt.show()
