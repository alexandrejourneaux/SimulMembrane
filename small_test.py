# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:15:59 2020

@author: alexa
"""

#%% Importations

import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from simulation_library import simulation as sm
from simulation_library.useful_functions import *




#%% Parameters definition -> Schnabel 2005
    
L_fc = 50e-2 #m
detuning = - 2 * np.pi * 15.15e6 #s-1
r1, r2 = np.sqrt(0.97), np.sqrt(0.9995)
t1, t2 = np.sqrt(1 - r1**2), np.sqrt(1 - r2**2)

#Frequency window observed    /!\ omega = detuning from the carrier (sidebands)
omega_min = 2 * np.pi * 12e6 #s-1
omega_max = 2 * np.pi * 18e6 #s-1 <- range of frequency sidebands observes
nb_freq = 1000 # <- freq resolution

omega = np.linspace(omega_min, omega_max, nb_freq)

squeezing_dB, squeezing_angle = 2, 0

filter_cavity_losses = 0
propagation_losses = 0
mode_mismatch_squeezer_filter_cavity = 0
mode_mismatch_squeezer_local_oscillator = 0
filter_cavity_length_noise = 0 # m -> Not used here 

injection_losses = 0 # à modifier -> pertes avant la fc
propagation_losses = 0 # à modifier -> pertes après la fc, avant l'ifo
readout_losses = 0 # à modifier -> pertes après l'ifo

phase_mm_default = 0



#%% Wigner functions at various frequencies (Schnabel)

       
freq = np.array([12, 13, 13.8, 14.1, 14.4, 14.7, 15.3, 15.6, 15.9, 16.2, 17, 18]) * (1e6)
subplt = 1

for f in freq:
    plt.subplot(4, 3, subplt)
    plt.axis('off')
    plt.title('{} MHz'.format(round(f / (1e5)) / 10))
    
    sqz = sm.Squeezer(5)
    fc = sm.ModeMismatchedFilterCavity(2*np.pi*f, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
    
    my_setup = sm.Setup([sqz, fc])

    state = sm.State()

    state.passesThroughSetup(my_setup)
    
    plt.imshow(pdf(np.real(state.covariance_matrix), 100))
    subplt += 1
