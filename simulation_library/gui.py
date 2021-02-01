# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:00:09 2021

@author: Alexandre
"""

import simulation_library.simulation as sm
from simulation_library.constants import c
import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
import pyqtgraph as pg
from pyinstruments import CurveDB
import os
import ast
from pathlib import Path

#Necessary line to save a curve in database
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi('simulation_library/maingui.ui', self)
    
        #Plot parameters
        self.freq_center = float(self.centerFreqBox.text())
        self.centerFreqButton.clicked.connect(self.setCenterFreq)
        
        self.freq_span = float(self.spanFreqBox.text())
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
        
        self.injection_losses = float(self.injectionLossesBox.text())
        self.injectionLossesButton.clicked.connect(self.setInjectionLosses)
        
        #Interferometer parameters
        self.ifo_length = float(self.ifoLengthBox.text())
        self.ifoLengthButton.clicked.connect(self.setIfoLength)

        self.ifo_finesse = float(self.ifoFinesseBox.text())
        self.ifoFinesseButton.clicked.connect(self.setIfoFinesse)
        
        self.m_eff = float(self.ifoMassBox.text())
        self.ifoMassButton.clicked.connect(self.setIfoMass)
        
        self.omega_m = 2 * np.pi * float(self.ifoOmegamBox.text())
        self.ifoOmegamButton.clicked.connect(self.setIfoOmegam)
        
        self.quality_factor = float(self.ifoQualityFactorBox.text())
        self.ifoQualityFactorButton.clicked.connect(self.setQualityFactor)
        
        self.propagation_losses = float(self.propagationLossesBox.text())
        self.propagationLossesButton.clicked.connect(self.setPropagationLosses)
        
        #Filter cavity 1 parameters
        self.detuning = 2 * np.pi * float(self.detuningBox.text())
        self.detuningButton.clicked.connect(self.setDetuning)
        
        self.input_transmission = float(self.inputTransmissionBox.text())
        self.inputTransmissionButton.clicked.connect(self.setInputTransmission)
        
        self.filter_cavity_losses = float(self.filterCavityLossesBox.text())
        self.filterCavityLossesButton.clicked.connect(self.setFilterCavityLosses)
        
        self.filter_cavity_length = float(self.filterCavityLengthBox.text())
        self.filterCavityLengthButton.clicked.connect(self.setFilterCavityLength)
        
        self.filter_cavity_finesse = 2 * np.pi / (self.input_transmission**2 + self.filter_cavity_losses)
        self.filterCavityFinesseBox.setText(str(int(self.filter_cavity_finesse)))
        self.filter_cavity_finesse = float(self.filterCavityFinesseBox.text())
        self.filterCavityFinesseButton.clicked.connect(self.setFilterCavityFinesse)
        
        self.filter_cavity_BW = c / (2 * self.filter_cavity_length * self.filter_cavity_finesse)
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        self.filter_cavity_BW = float(self.filterCavityBWBox.text())
        self.filterCavityBWButton.clicked.connect(self.setFilterCavityBW)
        
        #Filter cavity 2 parameters
        self.detuning2 = 2 * np.pi * float(self.detuningBox2.text())
        self.detuningButton2.clicked.connect(self.setDetuning2)
        
        self.input_transmission2 = float(self.inputTransmissionBox2.text())
        self.inputTransmissionButton2.clicked.connect(self.setInputTransmission2)
        
        self.filter_cavity_losses2 = float(self.filterCavityLossesBox2.text())
        self.filterCavityLossesButton2.clicked.connect(self.setFilterCavityLosses2)
        
        self.filter_cavity_length2 = float(self.filterCavityLengthBox2.text())
        self.filterCavityLengthButton2.clicked.connect(self.setFilterCavityLength2)
        
        self.filter_cavity_finesse2 = 2 * np.pi / (self.input_transmission2**2 + self.filter_cavity_losses2)
        self.filterCavityFinesseBox2.setText(str(int(self.filter_cavity_finesse2)))
        self.filter_cavity_finesse2 = float(self.filterCavityFinesseBox2.text())
        self.filterCavityFinesseButton2.clicked.connect(self.setFilterCavityFinesse2)
        
        self.filter_cavity_BW2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_finesse2)
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        self.filter_cavity_BW2 = float(self.filterCavityBWBox2.text())
        self.filterCavityBWButton2.clicked.connect(self.setFilterCavityBW2)
        
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
        
        #Options
        self.unitBox.setChecked(True)
        self.m2Hz = self.unitBox.isChecked()
        self.unitBox.clicked.connect(self.setUnit)
        
        #Curve saving
        self.saveCurveButton.clicked.connect(self.saveCurve)
        
        #Parameters saving and loading
        self.saveParametersButton.clicked.connect(self.saveParameters)
        self.loadParametersButton.clicked.connect(self.loadParameters)
        
        self.graph.setBackground('w')
        self.pen1 = pg.mkPen(color=(255, 0, 0))
        self.pen2 = pg.mkPen(color=(0, 0, 0), style=QtCore.Qt.DashLine)
        
        self.setWindowTitle('SimulMembrane')
        
        self.label_style = {'color':'r', 'font-size':'15px'}
        self.graph.setLabel('left', 'Position noise (m²/Hz)', **self.label_style)
        self.graph.setLabel('bottom', 'Sideband frequency', 'Hz', **self.label_style)
        self.graph.addLegend()
        self.graph.setLogMode(False, True)
        
        self.current_curve = None
        
        self.plot()
        
        
        
    
    def setDetuning(self):
        self.detuning = 2 * np.pi * float(self.detuningBox.text())
        self.plot()
        
    def setDetuning2(self):
        self.detuning2 = 2 * np.pi * float(self.detuningBox2.text())
        self.plot()
    
    def setCenterFreq(self):
        self.freq_center = float(self.centerFreqBox.text())
        self.plot()
        
    def setSpanFreq(self):
        self.freq_span = float(self.spanFreqBox.text())
        self.plot()
    
    def setInputTransmission(self):
        self.input_transmission = float(self.inputTransmissionBox.text())
        
        self.filter_cavity_finesse = 2 * np.pi / (self.filter_cavity_losses + self.input_transmission**2)
        self.filterCavityFinesseBox.setText(str(int(self.filter_cavity_finesse)))
        
        self.filter_cavity_BW = c / (2 * self.filter_cavity_length * self.filter_cavity_finesse)
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        
        self.plot()
    
    def setInputTransmission2(self):
        self.input_transmission2 = float(self.inputTransmissionBox2.text())
        
        self.filter_cavity_finesse2 = 2 * np.pi / (self.filter_cavity_losses2 + self.input_transmission2**2)
        self.filterCavityFinesseBox2.setText(str(int(self.filter_cavity_finesse2)))
        
        self.filter_cavity_BW2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_finesse2)
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        
        self.plot()
    
    def setFilterCavityLosses(self):
        self.filter_cavity_losses = float(self.filterCavityLossesBox.text())
        
        self.filter_cavity_finesse = 2 * np.pi / (self.filter_cavity_losses + self.input_transmission**2)
        self.filterCavityFinesseBox.setText(str(int(self.filter_cavity_finesse)))
        
        self.filter_cavity_BW = c / (2 * self.filter_cavity_length * self.filter_cavity_finesse)
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        
        self.plot()
    
    def setFilterCavityLosses2(self):
        self.filter_cavity_losses2 = float(self.filterCavityLossesBox2.text())
        
        self.filter_cavity_finesse2 = 2 * np.pi / (self.filter_cavity_losses2 + self.input_transmission2**2)
        self.filterCavityFinesseBox2.setText(str(int(self.filter_cavity_finesse2)))
        
        self.filter_cavity_BW2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_finesse2)
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        
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
        self.filter_cavity_length = float(self.filterCavityLengthBox.text())
        
        self.filter_cavity_BW = c / (2 * self.filter_cavity_length * self.filter_cavity_finesse)
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        
        self.plot()

    def setFilterCavityLength2(self):
        self.filter_cavity_length2 = float(self.filterCavityLengthBox2.text())
        
        self.filter_cavity_BW2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_finesse2)
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        
        self.plot()
    
    def setIfoLength(self):
        self.ifo_length = float(self.ifoLengthBox.text())
        self.plot()
        
    def setIfoFinesse(self):
        self.ifo_finesse = float(self.ifoFinesseBox.text())
        self.plot()
    
    def setIfoMass(self):
        self.m_eff = 1e-9 * float(self.ifoMassBox.text())
        self.plot()
    
    def setIfoOmegam(self):
        self.omega_m = 2 * np.pi * float(self.ifoOmegamBox.text())
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
        
    def setFilterCavityFinesse(self):
        self.filter_cavity_finesse = float(self.filterCavityFinesseBox.text())
        
        self.filter_cavity_BW = c / (2 * self.filter_cavity_length * self.filter_cavity_finesse)
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        
        self.filter_cavity_losses = 2 * np.pi / self.filter_cavity_finesse - self.input_transmission**2
        self.filterCavityLossesBox.setText(str(int(self.filter_cavity_losses * 1000) / 1000))
        
        self.plot()
        
    def setFilterCavityFinesse2(self):
        self.filter_cavity_finesse2 = float(self.filterCavityFinesseBox2.text())
        
        self.filter_cavity_BW2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_finesse2)
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        
        self.filter_cavity_losses2 = 2 * np.pi / self.filter_cavity_finesse2 - self.input_transmission2**2
        self.filterCavityLossesBox2.setText(str(int(self.filter_cavity_losses2 * 1000) / 1000))
        
        self.plot()

    def setFilterCavityBW(self):
        self.filter_cavity_BW = float(self.filterCavityBWBox.text())
        
        self.filter_cavity_finesse = c / (2 * self.filter_cavity_length * self.filter_cavity_BW)
        self.filterCavityFinesseBox.setText(str(int(self.filter_cavity_finesse)))
        
        self.filter_cavity_losses = 2 * np.pi / self.filter_cavity_finesse - self.input_transmission**2
        self.filterCavityLossesBox.setText(str(int(self.filter_cavity_losses * 1000) / 1000))
        
        self.plot()
        
    def setFilterCavityBW2(self):
        self.filter_cavity_BW2 = float(self.filterCavityBWBox2.text())
        
        self.filter_cavity_finesse2 = c / (2 * self.filter_cavity_length2 * self.filter_cavity_BW2)
        self.filterCavityFinesseBox2.setText(str(int(self.filter_cavity_finesse2)))
        
        self.filter_cavity_losses2 = 2 * np.pi / self.filter_cavity_finesse2 - self.input_transmission2**2
        self.filterCavityLossesBox2.setText(str(int(self.filter_cavity_losses2 * 1000) / 1000))
        
        self.plot()

    def setUnit(self):
        self.m2Hz = self.unitBox.isChecked()
        self.graph.setLogMode(False, self.m2Hz)
        self.plot(rescale = True)
    
    def saveCurve(self):
        curve = CurveDB.create(*self.current_curve, name='SimulMembrane')
    
    def saveParameters(self):
        parameters = {'input_intensity' : self.input_intensity, \
                      'wavelength' : self.wavelength, \
                          'filter_cavity_length' : self.filter_cavity_length, \
                          'input_transmission' : self.input_transmission, \
                          'detuning' : self.detuning, \
                          'filter_cavity_finesse' : self.filter_cavity_finesse, \
                          'filter_cavity_BW' : self.filter_cavity_BW, \
                          'filter_cavity_length2' : self.filter_cavity_length2, \
                          'input_transmission2' : self.input_transmission2, \
                          'detuning2' : self.detuning2, \
                          'filter_cavity_finesse2' : self.filter_cavity_finesse2, \
                          'filter_cavity_BW2' : self.filter_cavity_BW2, \
                          'ifo_length' : self.ifo_length, \
                          'ifo_finesse' : self.ifo_finesse, \
                          'm_eff' : self.m_eff, \
                          'omega_m' : self.omega_m, \
                          'quality_factor' : self.quality_factor, \
                          'filter_cavity_losses' : self.filter_cavity_losses, \
                          'filter_cavity_losses2' : self.filter_cavity_losses2, \
                          'injection_losses' : self.injection_losses, \
                          'propagation_losses' : self.propagation_losses, \
                          'readout_losses' : self.readout_losses, \
                          'mode_mismatch_squeezer_filter_cavity' : self.mode_mismatch_squeezer_filter_cavity, \
                          'mode_mismatch_squeezer_local_oscillator' : self.mode_mismatch_squeezer_local_oscillator, \
                          'squeezing_factor' : self.squeezing_factor, \
                          'squeezing_angle' : self.squeezing_angle, \
                          'homodyne_angle' : self.homodyne_angle, \
                          'freq_center' : self.freq_center, \
                          'freq_span' : self.freq_span}
        file = open("parameters.txt","w")
        file.write(str(parameters))
        file.close()
    
    def loadParameters(self):
        # home_dir = str(Path.home())
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        
        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                loaded_parameters = ast.literal_eval(data)
                
                for p in loaded_parameters:
                    setattr(self, p, loaded_parameters[p])
                    # exec("self.{} = loaded_parameters[{}]".format(p, p))
                
                self.plot()
                self.updateBoxes()
    
    def updateBoxes(self):
        #Plot parameters
        self.centerFreqBox.setText("{:.2e}".format(self.freq_center))
        self.spanFreqBox.setText("{:.2e}".format(self.freq_span))
        self.inputIntensityBox.setText("{:.2e}".format(self.input_intensity))
        self.wavelengthBox.setText(str(1e9 * self.wavelength))
        self.squeezingFactorBox.setText(str(self.squeezing_factor))
        self.squeezingAngleBox.setText(str(180 / np.pi * self.squeezing_angle))
        self.injectionLossesBox.setText(str(self.injection_losses))
        self.detuningBox.setText("{:.2e}".format(self.detuning / (2 * np.pi)))
        self.inputTransmissionBox.setText(str(self.input_transmission))
        self.filterCavityLossesBox.setText(str(self.filter_cavity_losses))
        self.filterCavityLengthBox.setText(str(self.filter_cavity_length))
        self.filterCavityFinesseBox.setText(str(int(self.filter_cavity_finesse)))
        self.filterCavityBWBox.setText("{:.2e}".format(self.filter_cavity_BW))
        self.detuningBox2.setText("{:.2e}".format(self.detuning2 / (2 * np.pi)))
        self.inputTransmissionBox2.setText(str(self.input_transmission2))
        self.filterCavityLossesBox2.setText(str(self.filter_cavity_losses2))
        self.filterCavityLengthBox2.setText(str(self.filter_cavity_length2))
        self.filterCavityFinesseBox2.setText(str(int(self.filter_cavity_finesse2)))
        self.filterCavityBWBox2.setText("{:.2e}".format(self.filter_cavity_BW2))
        self.ifoLengthBox.setText("{:.2e}".format(self.ifo_length))
        self.ifoFinesseBox.setText(str(int(self.ifo_finesse)))
        self.ifoMassBox.setText("{:.2e}".format(self.m_eff))
        self.ifoOmegamBox.setText("{:.2e}".format(self.omega_m / (2 * np.pi)))
        self.ifoQualityFactorBox.setText("{:.2e}".format(self.quality_factor))
        self.propagationLossesBox.setText(str(self.propagation_losses))
        self.homodyneAngleBox.setText(str(180 / np.pi * self.homodyne_angle))
        self.readoutLossesBox.setText(str(self.readout_losses))
        self.mmSqueezerCavityBox.setText(str(self.mode_mismatch_squeezer_filter_cavity))
        self.mmSqueezerLOBox.setText(str(self.mode_mismatch_squeezer_local_oscillator))
        

    def plot(self, rescale = False):
        '''Plots the noise spectrum of the output of the setup defined in var() and shot()'''
        
        #Laser
        intensity_input = self.input_intensity # photons/s
        lambda_carrier = self.wavelength # m
        omega_carrier = 2 * np.pi * c / lambda_carrier
        
        # Filter cavity 1
        L_fc = self.filter_cavity_length # m
        t1 = self.input_transmission
        detuning = self.detuning #rad/s
        phase_mm_default = np.pi  # worst-case scenario for the phase experienced by the mode mismatched field upon reflection on the filter cavity
        
        # Filter cavity 2
        L_fc2 = self.filter_cavity_length2 # m
        t12 = self.input_transmission2
        detuning2 = self.detuning2 #rad/s
        phase_mm_default = np.pi  # worst-case scenario for the phase experienced by the mode mismatched field upon reflection on the filter cavity
        
        # Interferometer (membrane cavity)
        L_ifo = self.ifo_length # m
        finesse = self.ifo_finesse
        t_in = np.sqrt(2 * np.pi / finesse)
        m_eff = self.m_eff # kg
        omega_m = self.omega_m # rad/s
        Q = self.quality_factor
        
        gamma = t_in**2 / 2
        
        # Parameters to be measured for our experiment
        filter_cavity_losses = self.filter_cavity_losses
        filter_cavity_losses2 = self.filter_cavity_losses2
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
            '''var(omega) returns the noise at freq omega with squeezed light'''
            
            sqz = sm.Squeezer(squeezing_dB, squeezing_angle)
            injection = sm.Losses(injection_losses)
            ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
            propagation = sm.Losses(propagation_losses)
            fc1 = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            fc2 = sm.ModeMismatchedFilterCavity(omega, detuning2, L_fc2, t12, filter_cavity_losses2, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            readout = sm.Losses(readout_losses)
            
            my_setup = sm.Setup([sqz, injection, ifo, propagation, fc1, fc2, readout])
            
            state = sm.State()
            
            state.passesThroughSetup(my_setup)
            
            if self.m2Hz:
                return state.Sxx(homodyne_angle, omega, lambda_carrier, finesse, intensity_input, omega_m)
            
            else:
                return state.variance(homodyne_angle)
        
        def shot(omega):
            '''shot(omega) returns the noise at freq omega with only vacuum fluctuation in the input port'''
            
            injection = sm.Losses(injection_losses)
            ifo = sm.Interferometer(omega, omega_m, m_eff, gamma, L_ifo, lambda_carrier, t_in, intensity_input, Q)
            propagation = sm.Losses(propagation_losses)
            fc1 = sm.ModeMismatchedFilterCavity(omega, detuning, L_fc, t1, filter_cavity_losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            fc2 = sm.ModeMismatchedFilterCavity(omega, detuning2, L_fc2, t12, filter_cavity_losses2, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm_default)
            readout = sm.Losses(readout_losses)
            
            my_setup = sm.Setup([injection, ifo, propagation, fc1, fc2, readout])
            
            state = sm.State()
            
            state.passesThroughSetup(my_setup)
            
            if self.m2Hz:
                self.graph.setLabel('left', 'Position noise (m²/Hz)', **self.label_style)
                return state.Sxx(homodyne_angle, omega, lambda_carrier, finesse, intensity_input, omega_m)
            
            else:
                self.graph.setLabel('left', 'Phase noise (dB)', **self.label_style)
                return state.variance(homodyne_angle)
        
        self.graph.clear()
        
        omega_min = 2 * np.pi * freq_min #s-1
        omega_max = 2 * np.pi * freq_max #s-1 <- range of sideband frequencies observed
        nb_freq = 1000 # <- freq resolution²
        
        omega_array = np.linspace(omega_min, omega_max, nb_freq)
        
        noise = np.array([var(omega) for omega in omega_array])
        vac_noise = np.array([shot(omega) for omega in omega_array])
        
        if rescale:
            self.graph.setYRange(min([*noise, *vac_noise]), max([*noise, *vac_noise]), padding=0)
        
        self.graph.plot(omega_array / (2 * np.pi), noise, name = 'With squeezer', pen = self.pen1)
        self.graph.plot(omega_array / (2 * np.pi), vac_noise, name = 'Without squeezer', pen = self.pen2)
        
        self.current_curve = omega_array, noise


