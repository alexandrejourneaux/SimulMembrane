# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:32:02 2021

@author: Alexandre
"""

import numpy as np
from simulation_library.constants import *
from simulation_library import simulation as sm
import matplotlib.pyplot as plt
# from simulation_library import gui

#%% Parameters definition

intensity_input = 5e15 # photons/s
lambda_carrier = 1064e-9 # m
omega_carrier = 2 * np.pi * c / lambda_carrier

# Filter cavity
L_fc = 85e-2 # m
t1, t2 = 0.07, 0
detuning = 2 * np.pi * 1.5e6 #Hz

# Interferometer (membrane cavity)
L_ifo = 1e-4 # m
finesse = 50e3
t_in = np.sqrt(2 * np.pi / finesse)
m_eff = 100e-12 # kg
omega_m = 2 * np.pi * 1.5e6 # Hz
Q = 1e7

gamma = t_in**2 / 2
tau = 2 * L_ifo / c
omega_cav = gamma / tau

# Paper parameters, to be measured for our experiment
filter_cavity_losses = 0
injection_losses = 0        # à modifier -> pertes avant la fc
propagation_losses = 0      # à modifier -> pertes après la fc, avant l'ifo
readout_losses = 0          # à modifier -> pertes après l'ifo
mode_mismatch_squeezer_filter_cavity = 0.0
mode_mismatch_squeezer_local_oscillator = 0.0

# Squeezing factor and squeezed quadrature
squeezing_dB = 5
squeezing_angle = np.pi/2 # dB, rad

phase_mm_default = np.pi  # worst-case scenario ?

# Window of simulated frequencies
freq_span = 1e6 #Hz
freq_min = omega_m / (2 * np.pi) - freq_span
freq_max = omega_m / (2 * np.pi) + freq_span

# omega = 0 #omega stands for the detuning to the carrier

#%% Simulation at one frequency

# omega = 0

# sqz = sm.Squeezer(10)
# injection = sm.Losses(0.36)
# ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
# fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)

# my_setup = sm.Setup([sqz, injection, ifo, fc])

# state = sm.State()

# state.passesThroughSetup(my_setup)

#%% Plot, frequency-dependant

def plot():
    
    quadrature_angle_homodyne = np.pi/2
    
    def var(omega):
        sqz = sm.Squeezer(10, np.pi/2)
        injection = sm.Losses(0)
        ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
        fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
        
        my_setup = sm.Setup([sqz, injection, fc, ifo])
        
        state = sm.State()
        
        state.passesThroughSetup(my_setup)
        
        return state.variance(quadrature_angle_homodyne)
    
    def shot(omega):
        injection = sm.Losses(0)
        ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
        fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
        
        my_setup = sm.Setup([injection, fc, ifo])
        
        state = sm.State()
        
        state.passesThroughSetup(my_setup)
        
        return state.variance(quadrature_angle_homodyne)
        
    
    freq_min = 0
    freq_max = 3e6
    
    omega_min = 2 * np.pi * freq_min #s-1
    omega_max = 2 * np.pi * freq_max #s-1 <- range of sideband frequencies observed
    nb_freq = 1000 # <- freq resolution²
    
    omega_array = np.linspace(omega_min, omega_max, nb_freq)
    
    noise = [var(omega) for omega in omega_array]
    vac_noise = [shot(omega) for omega in omega_array]
    
    plt.plot(omega_array / (2 * np.pi), noise)
    plt.plot(omega_array / (2 * np.pi), vac_noise)
    


plot()




