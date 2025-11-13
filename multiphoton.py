import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

#analyse FFT
factors = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0017617591610262604, 0.0018395818402202864, 0.001521805016805785, 0.0013803610826568773, 0.0016446185653961596, 0.000720659336646127], [0.003358757038226921, 0.002012818036050654, 0.0014348666147423767, 0.0006796583034160667]]
path = Path("triple")  # Change to desired path

height_fit = {"87":[2.2810541, 0.07898855], "85":[2.1531674, 0.16661867], "triple":[2.1531674, 0.16661867]}
def height_model(x, A, k):
    x = np.asarray(x, dtype=float)
    return k*(x**2)/(1+A*(x**2))


def sorting_key(file_name):
    name = file_name.split(".")[0].split("-")
    if len(name) > 1:
        return int(name[0]) + int(name[1])/100
    else:
        return int(name[0])

files = sorted([f.name for f in path.glob("*.csv")], key=sorting_key, reverse=True)  # Sort by file name
start_time = 7 #s
end_time = 20 #s
print(files)

gains = {"triple": [1000, 1000, 1000, 1000_000],
         "85": [1000, 1000,1000,1000,1000,1000],
         "87": [1000,1000, 1000,1000,1000,1000]}

def peak_profile(x, height0, width0, pos0, offset, drift): #Lorentzian profile
    return -abs(height0) / (1 + ((x - pos0) / width0) ** 2) + offset + drift*x  #+extra_drift*x**2

def power_law(x, a, b):
    return a * x**b 

#all the gains 1000
peaks = []
heights = []
heights_err = []
for k, power in enumerate(files):
    df = pd.read_csv(path / power, skiprows=2).apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    mask = (df[:, 0] >= start_time) & (df[:, 0] <= end_time)
    df = df[mask]
    df[:,1] = df[:, 1]/gains[path.name][k]
    if len(peaks) == 0:
        peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.001/1000, distance=2000, prominence=0.001/1000, width=2000)
        #peaks, properties = scipy.signal.find_peaks(-df[:, 1], height=0.01, distance=1000, prominence=0.02, width=200)
    else:
        peaks = [np.argmin(df[:,1])]
    
    plt.plot(df[:, 0], df[:, 1], label="Data")
    for i, peak in enumerate(peaks):
        try:
            if False:
                window = 30_000
                df = df[peak-window:peak+window,:]
                peak = window
            popt, pcov = curve_fit(lambda x, height, width, pos, off, drift: peak_profile(x, height, width, pos, off, drift), df[:, 0], df[:, 1], p0=[2*df[peak, 1], df[peak, 1]/3, df[peak, 0], 0, 0])
            plt.plot(df[:, 0], peak_profile(df[:, 0], *popt), label=f"Fit {peak}")
            plt.vlines(df[peak, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peak}")
            plt.vlines(df[min(peak, df.shape[0]-1), 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')

        except (RuntimeError, ValueError) as e:
            plt.close()
            plt.vlines(df[peak, 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed', label=f"Peak {peak}")
            plt.vlines(df[min(peak, df.shape[0]-1), 0], ymin=np.min(df[:, 1]), ymax=np.max(df[:, 1]), colors='r', linestyles='dashed')
            plt.plot(df[:, 0], df[:, 1], label="Data")
            plt.show()
            print(f"Could not fit {power}: {e}")
            continue
        heights.append(abs(popt[0]))
        heights_err.append(abs(popt[0])*0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [V]")
    plt.title(f"Drive: {power.split('.')[0]} V")
    plt.legend()
    plt.show()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 20,
    "font.size": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

colours = {"85": "green", "87":"red", "triple": "blue" }

labels = {"87":r"$^{87}\mathrm{Rb}$",
    "85":r"$^{85}\mathrm{Rb}$",
    "triple":"Three photon excitation"
}

voltages = [sorting_key(f) for f in files]
coeffs, cov = np.polyfit(np.log(voltages), np.log(heights), 1, cov=True)
fit_line = np.poly1d(coeffs)
plt.plot(voltages, np.exp(fit_line(np.log(voltages))), label=rf"{labels[path.name]}: $\alpha = {round(coeffs[0], 1)} \pm {round(np.sqrt(cov[0][0]), 1)}$", linestyle="--", color = colours[path.name])

simulated_values = height_model(voltages, *height_fit[path.name])*np.array(factors[2])
plt.errorbar(voltages, heights, heights_err, marker="+",  color = colours[path.name], alpha = 0.6, linestyle = "none")
plt.fill_between(
    voltages,
    simulated_values,
    color=colours[path.name],
    alpha=0.30,
    label=f"Noise {labels[path.name]}"
)
plt.xlabel("Forcing [V]")
plt.xscale("log")
plt.minorticks_off()
plt.xticks(voltages, [f"{t:g}" for t in voltages])
plt.yscale("log")
plt.ylabel(f"Height [V]")
plt.title(f"Three photon excitation - Peak height vs Forcing")
plt.legend()
plt.tight_layout()
plt.grid(which="both", alpha = 0.3)
plt.show()



#power law

double_85 = [0.003159364718945046, 0.00192278414227056, 0.001186341539450881, 0.0007008656055881963, 0.0003889840259912829, 0.00019819796375885113]
background_85 = [1.36172198e-04, 1.42148701e-04, 1.17548665e-04, 1.06564048e-04, 1.26856468e-04, 5.55090189e-05]
err_85 = [1.36172198e-04, 1.42148701e-04, 1.17548665e-04, 1.06564048e-04, 1.26856468e-04, 5.55090189e-05]

double_87 = [0.0008324853516125386, 0.000498274906783401, 0.0003298512242858734, 0.00021003307330352217, 0.00014649572112397462, 9.187592311672687e-05]
err_87 = [9.973077653458308e-05, 6.182415225065895e-05, 4.056409243084761e-05, 2.6356057023487088e-05, 1.806685456097309e-05, 1.14417617579328e-05]
err_87 = [2.0*n for n in err_87]
background_87 = [6.09395685e-05, 6.36151327e-05, 5.26071149e-05, 4.76925956e-05, 5.67771497e-05, 2.48461335e-05]

paths = [Path("double/85"), Path("double/87")]

heights = {"85": double_85, "87":double_87}
heights_err = {"85": err_85, "87": err_87}

for path in paths:
    coeffs, cov = np.polyfit(np.log(voltages), np.log(heights[path.name]), 1, cov=True)
    fit_line = np.poly1d(coeffs)
    if path.name == "87":
           plt.plot(voltages, np.exp(fit_line(np.log(voltages))), label=rf"{labels[path.name]}: $\alpha = {round(coeffs[0], 1)} \pm {0.6}$", linestyle="--", color = colours[path.name])
    else:      
        plt.plot(voltages, np.exp(fit_line(np.log(voltages))), label=rf"{labels[path.name]}: $\alpha = {round(coeffs[0], 1)} \pm {round(np.sqrt(cov[0][0]), 1)}$", linestyle="--", color = colours[path.name])

    simulated_values = height_model(voltages, *height_fit[path.name])*np.array(factors[1])
    plt.errorbar(voltages, heights[path.name], heights_err[path.name], marker="+",  color = colours[path.name], alpha = 0.6, linestyle = "none")
    plt.fill_between(
        voltages,
        simulated_values,
        color=colours[path.name],
        alpha=0.30,
        label=f"Noise {labels[path.name]}"
    )
plt.xlabel("Forcing [V]")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(f"Height [V]")
plt.title(f"Two photon excitation - Peak height vs Forcing")
plt.legend()
plt.grid(which="both", alpha = 0.3)
plt.show()



