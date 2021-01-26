# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:00:09 2021

@author: Alexandre
"""

import simulation_library.simulation as sm
from simulation_library.constants import c
import numpy as np
from PyQt5 import QtWidgets, uic


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi('simulation_library/maingui.ui', self)
    
        #Plot parameters
        self.freq_center = 1e6 * float(self.centerFreqBox.text())
        self.centerFreqButton.clicked.connect(self.setCenterFreq)
        
        self.freq_span = 1e6 * float(self.spanFreqBox.text())
        self.spanFreqButton.clicked.connect(self.setSpanFreq)
        
        #Laser parameters
        self.input_intensity = float(self.inputIntensityBox.text())
        self.inputIntensityButton.clicked.connect(self.setInputIntensity)
        
        self.wavelength = 1e-9 * float(self.wavelengthBox.text())
        self.wavelengthButton.clicked.connect(self.setWavelength)
        
        #Squeezer parameters
        self.squeezing_factor = float(self.squeezingFactorBox.text())
        self.squeezingFactorButton.clicked.connect(self.setSqueezingFactor)
        
        self.squeezing_angle = np.pi / 180 * float(self.squeezingAngleBox.text())
        self.squeezingAngleButton.clicked.connect(self.setSqueezingAngle)
        
        self.injection_losses = np.pi / 180 * float(self.injectionLossesBox.text())
        self.injectionLossesButton.clicked.connect(self.setInjectionLosses)
        
        #Filter cavity parameters
        self.detuning = 2 * np.pi * 1e6 * float(self.detuningBox.text())
        self.detuningButton.clicked.connect(self.setDetuning)
        
        self.input_transmission = float(self.inputTransmissionBox.text())
        self.inputTransmissionButton.clicked.connect(self.setInputTransmission)
        
        self.filter_cavity_losses = float(self.filterCavityLossesBox.text())
        self.filterCavityLossesButton.clicked.connect(self.setFilterCavityLosses)
        
        self.filter_cavity_length = 1e-2 * float(self.filterCavityLengthBox.text())
        self.filterCavityLengthButton.clicked.connect(self.setFilterCavityLength)
        
        #Interferometer parameters
        self.ifo_length = 1e-6 * float(self.ifoLengthBox.text())
        self.ifoLengthButton.clicked.connect(self.setIfoLength)

        self.ifo_finesse = float(self.ifoFinesseBox.text())
        self.ifoFinesseButton.clicked.connect(self.setIfoFinesse)
        
        self.m_eff = 1e-9 * float(self.ifoMassBox.text())
        self.ifoMassButton.clicked.connect(self.setIfoMass)
        
        self.omega_m = 2 * np.pi * 1e6 * float(self.ifoOmegamBox.text())
        self.ifoOmegamButton.clicked.connect(self.setIfoOmegam)
        
        self.quality_factor = float(self.ifoQualityFactorBox.text())
        self.ifoQualityFactorButton.clicked.connect(self.setQualityFactor)
        
        self.propagation_losses = float(self.propagationLossesBox.text())
        self.propagationLossesButton.clicked.connect(self.setPropagationLosses)
        
        #Read-out parameters
        self.homodyne_angle = np.pi / 180 * float(self.homodyneAngleBox.text())
        self.homodyneAngleButton.clicked.connect(self.setHomodyneAngle)
        
        self.readout_losses = float(self.readoutLossesBox.text())
        self.readoutLossesButton.clicked.connect(self.setReadoutLosses)
        
        #Mode-mismatch parameters
        self.mode_mismatch_squeezer_filter_cavity = float(self.mmSqueezerCavityBox.text())
        self.mmSqueezerCavityButton.clicked.connect(self.setMmSqueezerCavity)
        
        self.mode_mismatch_squeezer_local_oscillator = float(self.mmSqueezerLOBox.text())
        self.mmSqueezerLOButton.clicked.connect(self.setMmSqueezerLO)
                
        self.plot()
        
    
    def setDetuning(self):
        self.detuning = 2 * np.pi * float(self.detuningBox.text()) * 1e6
        self.plot()
        
    def setCenterFreq(self):
        self.freq_center = 1e6 * float(self.centerFreqBox.text())
        self.plot()
        
    def setSpanFreq(self):
        self.freq_span = 1e6 * float(self.spanFreqBox.text())
        self.plot()
    
    def setInputTransmission(self):
        self.input_transmission = float(self.inputTransmissionBox.text())
        self.plot()
        
    def setFilterCavityLosses(self):
        self.filter_cavity_losses = float(self.filterCavityLossesBox.text())
        self.plot()
        
    def setHomodyneAngle(self):
        self.homodyne_angle = np.pi / 180 * float(self.homodyneAngleBox.text())
        self.plot()
        
    def setInputIntensity(self):
        self.input_intensity = float(self.inputIntensityBox.text())
        self.plot()
        
    def setWavelength(self):
        self.wavelength = 1e-9 * float(self.wavelengthBox.text())
        self.plot()
        
    def setFilterCavityLength(self):
        self.filter_cavity_length = 1e-2 * float(self.filterCavityLengthBox.text())
        self.plot()
    
    def setIfoLength(self):
        self.ifo_length = 1e-6 * float(self.ifoLengthBox.text())
        self.plot()
        
    def setIfoFinesse(self):
        self.ifo_finesse = float(self.ifoFinesseBox.text())
        self.plot()
    
    def setIfoMass(self):
        self.m_eff = 1e-9 * float(self.ifoMassBox.text())
        self.plot()
    
    def setIfoOmegam(self):
        self.omega_m = 2 * np.pi * 1e6 * float(self.ifoOmegamBox.text())
        self.plot()
    
    def setQualityFactor(self):
        self.quality_factor = float(self.ifoQualityFactorBox.text())
        self.plot()
        
    def setSqueezingFactor(self):
        self.squeezing_factor = float(self.squeezingFactorBox.text())
        self.plot()
    
    def setSqueezingAngle(self):
        self.squeezing_angle = np.pi / 180 * float(self.squeezingAngleBox.text())
        self.plot()
    
    def setInjectionLosses(self):
        self.injection_losses = float(self.injectionLossesBox.text())
        self.plot()
    
    def setPropagationLosses(self):
        self.propagation_losses = float(self.propagationLossesBox.text())
        self.plot()
    
    def setReadoutLosses(self):
        self.readout_losses = float(self.readoutLossesBox.text())
        self.plot()
    
    def setMmSqueezerCavity(self):
        self.mode_mismatch_squeezer_filter_cavity = float(self.mmSqueezerCavityBox.text())
        self.plot()
    
    def setMmSqueezerLO(self):
        self.mode_mismatch_squeezer_local_oscillator = float(self.mmSqueezerLOBox.text())
        self.plot()

    
    def plot(self):
        
        #Laser
        intensity_input = self.input_intensity # photons/s
        lambda_carrier = self.wavelength # m
        omega_carrier = 2 * np.pi * c / lambda_carrier
        
        # Filter cavity
        L_fc = self.filter_cavity_length # m
        t1 = self.input_transmission
        detuning = self.detuning #rad/s
        phase_mm_default = np.pi  # worst-case scenario for the phase experienced by the mode mismatched filed upon reflection on the filter cavity
        
        # Interferometer (membrane cavity)
        L_ifo = self.ifo_length # m
        finesse = self.ifo_finesse
        t_in = np.sqrt(2 * np.pi / finesse)
        m_eff = self.m_eff # kg
        omega_m = self.omega_m # rad/s
        Q = self.quality_factor
        
        gamma = t_in**2 / 2
        tau = 2 * L_ifo / c
        omega_cav = gamma / tau
        
        # Parameters to be measured for our experiment
        filter_cavity_losses = self.filter_cavity_losses
        injection_losses = self.injection_losses
        propagation_losses = self.propagation_losses
        readout_losses = self.readout_losses
        mode_mismatch_squeezer_filter_cavity = self.mode_mismatch_squeezer_filter_cavity
        mode_mismatch_squeezer_local_oscillator = self.mode_mismatch_squeezer_local_oscillator
        
        # Squeezer
        squeezing_dB = self.squeezing_factor
        squeezing_angle = self.squeezing_angle # dB, rad
        
        #Readout
        homodyne_angle = self.homodyne_angle
        
        #Plot
        freq_min = self.freq_center - self.freq_span
        freq_max = self.freq_center + self.freq_span
        

        
        def var(omega):
            sqz = sm.Squeezer(squeezing_dB, squeezing_angle)
            injection = sm.Losses(injection_losses)
            fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            propagation = sm.Losses(propagation_losses)
            ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
            readout = sm.Losses(readout_losses)
            
            my_setup = sm.Setup([sqz, injection, fc, propagation, ifo, readout])
            
            state = sm.State()
            
            state.passesThroughSetup(my_setup)
            
            return state.variance(homodyne_angle)
        
        def shot(omega):
            injection = sm.Losses(injection_losses)
            fc = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            propagation = sm.Losses(propagation_losses)
            ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
            readout = sm.Losses(readout_losses)
            
            my_setup = sm.Setup([injection, fc, propagation, ifo, readout])
            
            state = sm.State()
            
            state.passesThroughSetup(my_setup)
            
            return state.variance(homodyne_angle)
        
        omega_min = 2 * np.pi * freq_min #s-1
        omega_max = 2 * np.pi * freq_max #s-1 <- range of sideband frequencies observed
        nb_freq = 1000 # <- freq resolution²
        
        omega_array = np.linspace(omega_min, omega_max, nb_freq)
        
        noise = [var(omega) for omega in omega_array]
        vac_noise = [shot(omega) for omega in omega_array]
        
        self.graph.clear()
        
        self.graph.plot(omega_array / (2 * np.pi), noise)
        self.graph.plot(omega_array / (2 * np.pi), vac_noise)


