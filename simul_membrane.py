# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:32:02 2021

@author: Alexandre
"""

from simulation_library import gui
from PyQt5 import QtWidgets
import sys

#from simulation_library import simulation as sm

#%% Plot, frequency-dependant

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = gui.MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()

#%% Example code for simulation at one frequency

# omega = 0

# sqz = sm.Squeezer(10)
# injection = sm.Losses(0.36)
# ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
# fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)

# my_setup = sm.Setup([sqz, injection, ifo, fc])

# state = sm.State()

# state.passesThroughSetup(my_setup)











