Some notes about the measurements:

e1 - 17/10
    Time sweep 10s
    Gradient for B field signal 2.79 V/s
    Max Range
    Frequency in #kHz
Results:
Small peaks - I = 3/2 - Rb87
Big Peaks - I = 5/2 - Rb85
Did not cancel out magnetic field properly

e2 - 20/10
    Rotated the frame, now it is good ig (second data)
    Same settings
    0 ohm for horizontal + sweep at 4.674 = 0.281 G 
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
    Using 90kHz so it fits on the screen, two sets of measurement (wrong point)
    I think height saturates and width is non-zero at zero
    Slighlty worried about the time constant on the device and the shape of the peaks
    Well, the graph with the power broadening is wrong...
    You can calibrate magnetic field with the known gf
    Looking at double photon - tuned 100kHz, Gain 1000 (max), what should I measure??
    Voltage B-sweep magnitude 0.750-0.792 V, Horizontal off (0.001), time of sweep 10s
    For double photons, maybe I should set it to 100kHz, and then look for it
    I checked the second harmonics, documents
    
    Maxed gain for small ones (0.1, 0.2), half of the max gain (0.4) 50s sweep
    Go 100s sweep, max gain/5 = 200, for 0.8 and 1.6
    Figured what was wrong with the power broadening - need to have time constant the same for the whole time + gain the same
    Go with time of sweep 20s and whatever gain is convenient
    Add one more point for 50mV (gradient is going to be 2 then) Max gain, time constant 1s, 50s sweep time
    I think I can do better with the power broadening


Rabi - 28/10
    Need to finish for Rb87
    50kHz, 100gain, looking at the positive, 10Hz, so it "saturates"
    Noise is present event for the empty cable = it comes from the machine
    Repeat for Rb85, because the data for 87 are way too good
    50kHz, 20gain, positive, 10Hz

Power - 28/10
    Speedrun, focus on one, record gain and timesweep
    3s time constant, 150kHz, so I do not see background
    Constant drift up
    Graphs looks good, be carefull about the units
    The amplitude was changing because of detuning, and the lifetime changes because of the dephasing or whatever (decreasing)
    87 works better because SNR is better (HIGHER GAIN). The noise is coming from the machine
    Idk how to rid of the detuning

Last day - 29/10
    Doing the last peak broadening (small 87), 50s sweep time, 150kHz, 3 s time constant
    For small 0-02 value, we used 20s sweep time (drift is not that bad), 1 s time constant
    B sweep - 0.733 - 0.847 V?

    Rabi - idk which peak, 50.5 kHz (hopefully fine), gain 200

    Coils horizontal - distance 16.0 cm, diameter 31.1 cm, 
    Coils vertical - 22.7 cm diameter, 12 cm distance, 

    Double photon - Needed to go 200kHz and to avoid overlap, 0.857 - 1.034 V sweep coil and, sweep time 50s
    weird exponent
    Omg triple peaks - max gain, 20s sweep time, 300kHz, sweep field 0.623-0.454V, 85 ig
    

For power broadening the important thing is 85, 87 (full range)