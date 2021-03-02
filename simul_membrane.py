# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:32:02 2021

@author: Alexandre

The definition of the simulated setup is in the gui.py file, in the plot() function
"""

from simulation_library import gui
from PyQt5 import QtWidgets
import sys

#from simulation_library import simulation as sm

#%% GUI launcher

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = gui.MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()

#%% Example code for simulation at one frequency

# sqz = sm.Squeezer(squeezing_dB, squeezing_angle)
# injection = sm.Losses(injection_losses)
# fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
# propagation = sm.Losses(propagation_losses)
# ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
# readout = sm.Losses(readout_losses)

# my_setup = sm.Setup([sqz, injection, fc, propagation, ifo, readout])

# state = sm.State()

# state.passesThroughSetup(my_setup)











