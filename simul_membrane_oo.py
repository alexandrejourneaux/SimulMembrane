# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:32:02 2021

@author: Alexandre
"""

import numpy as np
from simulation_library.constants import *
from simulation_library import simulation as sm

#%% Parameters definition

intensity_input = 5e15 # photons/s
lambda_carrier = 1064e-9 # m
omega_carrier = 2 * np.pi * c / lambda_carrier

# Filter cavity
L_fc = 85e-2 # m
t1, t2 = 0.07, 0
detuning = 0

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

omega = 0 #omega stands for the detuning to the carrier

#%% Simulation

sqz = sm.Squeezer(10)
ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)

cov = sm.CovarianceMatrix()
cov.passes_through(sqz)
cov.passes_through(ifo)
cov.passes_through(fc)