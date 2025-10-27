Some notes about the measurements:

e1 - 17/10
    Time sweep 10s
    Gradient for B field signal 2.79 V/s
    Max Range
    Frequency in kHz
Results:
Small peaks - I = 3/2 - Rb87
Big Peaks - I = 5/2 - Rb85
Did not cancel out magnetic field properly

e2 - 20/10
    Rotated the frame, now it is good ig (second data)
    Same settings
    0 ohm for horizontal + sweep at 4.674 = 0.281 
    0.265V for vertical = 0.159 G
    Total B = 0.32 G
    Straight through

Rabi - 21/10
    osc2 - 30.548 kHz Driving, bigger 85 peak, 131.5 Hz trigger
    decay2 - same
    complicated - same, 19.5Hz trigger
    Channel A - signal
    Channel B - gate
    oscillations - same drive, 50Hz, one wavefront
    20V depends on voltage
    Try - 31.028 kHz,2.033Hz 

Rabi - 22/10
    Leave 50KHz, 100Hz, 500us/div, 20V setting, effective 19.7V
    Made the code
Rabi - 23/10
    Plan: Different voltages, detuning
    Voltages: 2.5 V steps, 2.5 bit shady, interesting pattern with beats
    Offset: 50kHz baseline, steps of 1 kHz on both sides? (max offset 5), 10Hz and 500us window
    10Hz so I am sure it decays
    
Rabi - 24/10
    Data analysis, git repo init, making some plots
    the fitting depends on the period of time you give it - short=> not enough oscillations, long => overfits the flat tail, only decay
    playing around with the straight signal - signal 80mV, offset 3.5V, time constantintroduces offset

Power Broadening - 27/10
    Using 90kHz so it fits on the screen, two sets of measurement
    I think height saturates and width is non-zero at zero
    Slighlty worried about the time constant on the device and the shape of the peaks
    You can calibrate magnetic field with the known gf
    Looking at double photon - tuned 100kHz, Gain 1000 (max), what should I measure??
    Voltage B-sweep magnitude 0.750-0.792 V, Horizontal off (0.001), time of sweep 10s
    Well, the graph with the power broadening is wrong...
    For double photons, maybe I should set it to 100kHz, and then look for it
    I checked the second harmonics, documents
    Maxed gain for small ones (0.1, 0.2), half of the max gain (0.4) 50s sweep
    Go 100s sweep, max gain/5 = 200, for 0.8 and 1.6


