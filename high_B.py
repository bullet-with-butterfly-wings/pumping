#read the high_B.csv file and plot the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("high_B/87/6-00.csv", skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
t = df[:, 0]
signal = df[:, 1]
plt.plot(t, signal, label="3.00 MHz")
plt.xlabel("Time [s]")
plt.ylabel("Signal [a.u.]")
plt.title("High B field signal vs time at 6.50 MHz")

plt.legend()
plt.show()