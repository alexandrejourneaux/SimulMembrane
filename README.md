# SimulMembrane
Simulation of noise spectrum for optomechanical interferometer

## Installation

You need to install the following packages to run the programme :

```
pip install numpy pyqt5 pyqtgraph
```

In addition, the following packages are necessary to run pyinstruments' CurveDB package :

```
pip install django django-evolution tablib django-import-export guidata guiqwt
pip install --upgrade django
```

You need to use the PyQt5 branch of the pyinstruments project.

## Launch GUI

To launch the GUI, run the file simul_membrane.py with python :

```
python simul_membrane.py
```

## Change simulated setup

If you wish to simulate another setup, you need to change the content of the loop in the plot function in the gui file.
Example of simulation at a given omega (the loop calculates this for every omega) :

```
omega = 0

sqz = sm.Squeezer(squeezing_dB, squeezing_angle)
injection = sm.Losses(injection_losses)
fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
propagation = sm.Losses(propagation_losses)
ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
readout = sm.Losses(readout_losses)

my_setup = sm.Setup([sqz, injection, fc, propagation, ifo, readout])

state = sm.State()

state.passesThroughSetup(my_setup)

print(state.variance(homodyne_angle))
```

The default values of the parameters are set in the maingui.ui file and can be modified by hand or with Qt Designer.
Other parameters can be added programmatically inside the plot function.