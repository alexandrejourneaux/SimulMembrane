# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:23:51 2021

@author: Alexandre
"""

import numpy as np
import cmath as cm
from simulation_library.constants import *

class CovarianceMatrix:
    
    def __init__(self, covariance_matrix = np.array([[1, 0], [0, 1]])):
        self.covariance_matrix = covariance_matrix
        
    def loss(self, losses):
        self.covariance_matrix = losses * np.eye(2) + (1 - losses) * self.covariance_matrix
    
    def passes_through(self, optical_element):
        self.covariance_matrix = optical_element.transfer_function(self.covariance_matrix)
    
    

class OpticalElement:
    
    def __init__(self, transfer_function):
        self.transfer_function = transfer_function
        
    
class LinearOpticalElement(OpticalElement):
    
    def __init__(self, two_photon_matrix):
        
        transfer = lambda M: two_photon_matrix.dot(M).dot(np.transpose(np.conjugate(two_photon_matrix)))
        OpticalElement.__init__(self, transfer)
        
        self.two_photon_matrix = two_photon_matrix
    


class Squeezer(LinearOpticalElement):
    
    def __init__(self, squeezing_dB, theta = 0):
        r = - 0.5 * np.log(10**(-squeezing_dB/10))
        amplitude_squeezed = np.matrix( [[np.exp(-r), 0], [0, np.exp(r)]] )
        rotation = np.matrix([[np.cos(theta), np.sin(theta)], [- np.sin(theta), np.cos(theta)]])
        
        LinearOpticalElement.__init__(self, rotation.dot(amplitude_squeezed).dot(np.transpose(np.conjugate(rotation))))
    
        
    
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
        
        A2 = (1 / np.sqrt(2)) * np.matrix([[1, 1], [-i, i]])
        one_photon_matrix = np.matrix([[reflection_coefficient(omega, detuning, input_transmission, losses), 0], [0, np.conjugate(reflection_coefficient(-omega, detuning, input_transmission, losses))]])
        
        two_photon_matrix = A2.dot(one_photon_matrix).dot(np.transpose(np.conjugate(A2)))
        
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
            
        MM = abs(tmm) * np.matrix([[np.cos(cm.phase(tmm)), - np.sin(cm.phase(tmm))], [np.sin(cm.phase(tmm)), np.cos(cm.phase(tmm))]])
        
        def reflection(omega, detuning, input_transmission):
            '''gives the transfer function in field amplitude of the cavity'''
            
            epsilon = 2 * filter_cavity_losses / (input_transmission**2 + filter_cavity_losses)
            ksi = 4 * (omega - detuning) * L_fc / (c * (input_transmission**2 + filter_cavity_losses))
            
            return( 1 - (2 - epsilon) / (1 + i * ksi) )
        
        def transfer(cov_mat):
        
            cav = (t00 * FilterCavity(omega, detuning, length, input_transmission, losses).two_photon_matrix + MM).dot(cov_mat).dot(np.transpose(np.conjugate(t00 * FilterCavity(omega, detuning, length, input_transmission, losses).two_photon_matrix + MM)))
            cav_losses = 1 - (abs(t00 * FilterCavity(omega, detuning, length, input_transmission, losses).reflection_coefficient() + tmm)**2 + abs(t00 * FilterCavity(-omega, detuning, length, input_transmission, losses).reflection_coefficient() + tmm)**2) / 2
            cav_vac = cav_losses * CovarianceMatrix().covariance_matrix
            
            return cav + cav_vac
        
        OpticalElement.__init__(self, transfer)
        
        













