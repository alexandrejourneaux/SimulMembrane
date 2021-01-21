# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:23:51 2021

@author: Alexandre
"""

import numpy as np
import cmath as cm
from simulation_library.constants import *
from simulation_library.useful_functions import *

class CovarianceMatrix:
    
    def __init__(self, covariance_matrix = np.array([[1, 0], [0, 1]])):
        self.covariance_matrix = covariance_matrix
        
    def loss(self, losses):
        self.covariance_matrix = losses * np.eye(2) + (1 - losses) * self.covariance_matrix
    
    def passes_through(self, optical_element):
        self.covariance_matrix = optical_element.transfer_function(self.covariance_matrix)
    
    def variance(self, quadrature_angle = 0, dB = False):
        cov_rot = apply(rotation(-quadrature_angle), self.covariance_matrix)
        var = np.real(cov_rot[0, 0])
        if dB:
            return  10 * np.log10(var)
        else:
            return var
        
    def Sxx(self, theta, omega, lambda_carrier, finesse, intensity_input, omega_ifo):
        '''gives the measurement noise in m^2/Hz, taking into account the low-pass filter effect of the ifo'''
    
        return lambda_carrier / (256 * finesse**2 * intensity_input) * self.variance(theta) * (1 + (omega/omega_ifo)**2) 
    
    

class OpticalElement:
    
    def __init__(self, transfer_function):
        self.transfer_function = transfer_function
        
    
class LinearOpticalElement(OpticalElement):
    
    def __init__(self, two_photon_matrix):
        
        transfer = lambda M: apply(two_photon_matrix, M)
        OpticalElement.__init__(self, transfer)
        
        self.two_photon_matrix = two_photon_matrix
    


class Squeezer(LinearOpticalElement):
    
    def __init__(self, squeezing_dB, theta = 0):
        r = - 0.5 * np.log(10**(-squeezing_dB/10))
        amplitude_squeezed = np.matrix( [[np.exp(-r), 0], [0, np.exp(r)]] )
        
        LinearOpticalElement.__init__(self, apply(rotation(-theta), amplitude_squeezed))
    
        
    
class Interferometer(LinearOpticalElement):
    
    def __init__(self, omega, omega_m, m_eff, gamma, arm_length, lambda_carrier, input_transmission, input_intensity, mech_quality_factor):
        
        k = 2 * np.pi / lambda_carrier
        tau = 2 * arm_length / c
    
        chi = 1 / ( m_eff * (omega_m**2 - omega**2 - i * (omega_m * omega / mech_quality_factor) ) )

        diag = (gamma + (i * omega * tau)) / (gamma + (i * omega * tau))
        K = ( 16 * input_transmission * input_intensity * h_bar * k**2 / (gamma - i*omega*tau)**2 ) * chi
    
        LinearOpticalElement.__init__(self, np.array([[diag, 0], [K, diag]]))


class FilterCavity(LinearOpticalElement):
    
    def __init__(self, omega, detuning, length, input_transmission, losses):
        
        self.omega, self.detuning, self.length, self.input_transmission, self.losses = omega, detuning, length, input_transmission, losses
        
        def reflection_coefficient(omega, detuning, input_transmission, losses):
            '''gives the transfer function in field amplitude of the cavity'''
            
            epsilon = 2 * losses / (input_transmission**2 + losses)
            ksi = 4 * (omega - detuning) * length / (c * (input_transmission**2 + losses))
            
            return( 1 - (2 - epsilon) / (1 + i * ksi) )
        
        one_photon_matrix = np.matrix([[reflection_coefficient(omega, detuning, input_transmission, losses), 0], [0, np.conjugate(reflection_coefficient(-omega, detuning, input_transmission, losses))]])
        
        two_photon_matrix = apply(A2, one_photon_matrix)
        
        LinearOpticalElement.__init__(self, two_photon_matrix)
    
    def reflection_coefficient(self):
        '''gives the transfer function in field amplitude of the cavity'''
        
        epsilon = 2 * self.losses / (self.input_transmission**2 + self.losses)
        ksi = 4 * (self.omega - self.detuning) * self.length / (c * (self.input_transmission**2 + self.losses))
        
        return( 1 - (2 - epsilon) / (1 + i * ksi) )
    
        

class ModeMismatchedFilterCavity(OpticalElement):
    
    def __init__(self, omega, detuning, length, input_transmission, losses, mode_mismatch_squeezer_filter_cavity, mode_mismatch_squeezer_local_oscillator, phase_mm):
        a0 = np.sqrt(1 - mode_mismatch_squeezer_filter_cavity)
        c0 = np.sqrt(1 - mode_mismatch_squeezer_local_oscillator)
        b0 = a0 * c0 + np.exp(i * phase_mm) * np.sqrt((1 - a0**2) * (1 - c0**2))
        t00 = a0 * np.conjugate(b0)
        tmm = c0 - t00
            
        MM = abs(tmm) * rotation(cm.phase(tmm))
        
        def reflection(omega, detuning, input_transmission):
            '''gives the transfer function in field amplitude of the cavity'''
            
            epsilon = 2 * filter_cavity_losses / (input_transmission**2 + filter_cavity_losses)
            ksi = 4 * (omega - detuning) * L_fc / (c * (input_transmission**2 + filter_cavity_losses))
            
            return( 1 - (2 - epsilon) / (1 + i * ksi) )
        
        def transfer(cov_mat):
        
            cav = apply(t00 * FilterCavity(omega, detuning, length, input_transmission, losses).two_photon_matrix + MM, cov_mat)
            cav_losses = 1 - (abs(t00 * FilterCavity(omega, detuning, length, input_transmission, losses).reflection_coefficient() + tmm)**2 + abs(t00 * FilterCavity(-omega, detuning, length, input_transmission, losses).reflection_coefficient() + tmm)**2) / 2
            cav_vac = cav_losses * CovarianceMatrix().covariance_matrix
            
            return cav + cav_vac
        
        OpticalElement.__init__(self, transfer)
        
        













