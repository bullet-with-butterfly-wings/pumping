import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.signal import find_peaks

complete_harmonics = [[],[],[]]
f = 0.05
n = 3
# Get all voltage folders (assuming folder names are voltages)
voltage_folders = sorted([f for f in Path("double/fft").iterdir() if f.is_dir()], 
                    key=lambda x: float(x.name), reverse=True)

for voltage_folder in voltage_folders:
    voltage = float(voltage_folder.name)

    # Get all CSV files in this voltage folder
    csv_files = list(voltage_folder.glob("*.csv"))

    # Store harmonics for all files at this voltage
    voltage_harmonics = []
    for h in range(n):
        voltage_harmonics.append([])

    for csv_file in csv_files:
        # Read FFT data
        df = pd.read_csv(csv_file)
        col0_numeric = pd.to_numeric(df.iloc[1:, 0], errors='coerce')
        mask = col0_numeric < (n+1)*f
        df = df.iloc[1:].loc[mask]

        frequencies = df.iloc[:, 0].astype(float).values
        amplitudes  = df.iloc[:, 2].astype(float).values
 
        plt.plot(frequencies, amplitudes)
        plt.show()
        for h in range(n):
            # index of the closest frequency bin
            idx = np.argmin(np.abs(frequencies - (h+1)*f))
            voltage_harmonics[h].append(amplitudes[idx])
    for h in range(n):
        complete_harmonics[h].append(np.mean(voltage_harmonics[h]))

factors = [
    [10 ** ((o - complete_harmonics[0][i]) / 20) for i, o in enumerate(order)]
    for order in complete_harmonics
]
print(factors)