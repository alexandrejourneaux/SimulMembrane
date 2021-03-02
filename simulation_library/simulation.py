# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:23:51 2021

@author: Alexandre

Publications used for mathematical framework :
    PhysRevD.90.062006 : global framework
    PhysRevD.88.022002 : signal recycling
    
"""

import numpy as np
import cmath as cm
from simulation_library.constants import c, i, h_bar, A2
from simulation_library.useful_functions import rotation, apply


class State:
    '''Describes the state of the fluctuations of light at a given point'''
    
    def __init__(self, covariance_matrix = np.array([[1, 0], [0, 1]])):
        self.covariance_matrix = covariance_matrix
    
    def passesThrough(self, optical_element):
        self.covariance_matrix = optical_element.transfer_function(self.covariance_matrix)
        
    def passesThroughSetup(self, setup):
        
        for elt in setup.elements:            
            self.passesThrough(elt)
    
    def variance(self, quadrature_angle = 0, dB = True):
        cov_rot = apply(rotation(-quadrature_angle), self.covariance_matrix)
        var = np.real(cov_rot[0, 0])
        if dB:
            return  10 * np.log10(var)
        else:
            return var
        
    def Sxx(self, theta, omega, lambda_carrier, finesse, intensity_input, omega_ifo):
        '''gives the measurement noise in m^2/Hz, taking into account the low-pass filter effect of the ifo'''
    
        return lambda_carrier / (256 * finesse**2 * intensity_input) * self.variance(theta, False) * (1 + (omega/omega_ifo)**2) 
    
        

class OpticalElement:
    '''Class from which every optical element inherits'''
    
    def __init__(self, transfer_function):
        '''Optical elements are characterized by their input-output effect on the covariance matrix of the quantum noise'''
        
        self.transfer_function = transfer_function
        

class LinearOpticalElement(OpticalElement):
    '''Special case of lossless elements described by their two-photon matrix'''
    
    def __init__(self, two_photon_matrix):
        
        transfer = lambda M: apply(two_photon_matrix, M)
        OpticalElement.__init__(self, transfer)
        
        self.two_photon_matrix = two_photon_matrix
        

class Setup:
    '''Defines a sequence of optical elements (or losses) through which the beam passes'''
    
    def __init__(self, elements = list()):
        self.elements = elements
    
    def addElement(self, element):
        self.elements.append(element)
    
    def outputState(self, input_state = State()):
        '''Returns the output state when input_state is feeded to the setup without modifying this input state'''
        
        temp_state = copy.deepcopy(input_state)
        temp_state.passesThroughSetup(self)
        
        return temp_state
    

class Losses(OpticalElement):
    '''Injection of vacuum noise through a loss channel'''
    
    def __init__(self, losses):
        
        transfer = lambda M: losses * State().covariance_matrix + (1 - losses) * M
        OpticalElement.__init__(self, transfer)


class Squeezer(LinearOpticalElement):
    '''Transforms a vacuum fluctuations state into a squeezed state with a given squeeze angle and magnitude'''
    
    def __init__(self, squeezing_dB, theta = 0):
        r = - 0.5 * np.log(10**(-squeezing_dB/10))
        amplitude_squeezed = np.matrix( [[np.exp(-r), 0], [0, np.exp(r)]] )
        
        LinearOpticalElement.__init__(self, apply(rotation(-theta), amplitude_squeezed))
    
        
    
class Interferometer(LinearOpticalElement):
    
    def __init__(self, omega, omega_m, m_eff, finesse, arm_length, lambda_carrier, input_intensity, mech_quality_factor, signal_recycling_transmission = 1, L_recycling = 0):
        
        #Try with Kwee simulation paper:
        # signal_recycling_reflection = np.sqrt(1 - signal_recycling_transmission**2)
        # omega_carrier = 2 * np.pi * c / lambda_carrier
        # input_power = input_intensity * h_bar * omega_carrier
        
        # omega_SQL0 = 8 / c * np.sqrt(input_power * omega_carrier / m_eff)
        # omega_SQL = signal_recycling_transmission / (1 + signal_recycling_reflection) * omega_SQL0
        
        # gamma0 = c / (4 * arm_length * finesse) #arm cavity half-width
        # gamma = (1 + signal_recycling_reflection) / (1 - signal_recycling_reflection) * gamma0
        
        # K = omega_SQL**2 / (omega_m**2 - omega**2 - i * (omega_m * omega / mech_quality_factor)) * gamma**2 / (omega**2 + gamma**2)
        # print(K)
        
        # LinearOpticalElement.__init__(self, np.array([[1, 0], [K, 1]]))
        
        #Try with Kwee GW paper        
        # signal_recycling_reflection = np.sqrt(1 - signal_recycling_transmission**2)
        # omega_carrier = 2 * np.pi * c / lambda_carrier
        # gamma = c / (2 * arm_length * finesse)
        # input_power = input_intensity * h_bar * omega_carrier
        
        # chi = 1 / ( m_eff * (omega_m**2 - omega**2 - i * (omega_m * omega / mech_quality_factor) ) )
        # K = 8 * input_power * omega_carrier / (arm_length**2 * (omega**2 + gamma**2)) * chi
                
        # Phi = omega * L_recycling / c + np.arctan(2 * omega / gamma)
        # K_sr = K * signal_recycling_transmission**2 / abs(1 + np.exp(2 * i * Phi) * signal_recycling_reflection)**2

        # LinearOpticalElement.__init__(self, np.array([[1, 0], [K_sr, 1]]))
        
        #Try with own computations
        k = 2 * np.pi / lambda_carrier
        tau = 2 * arm_length / c
        t_in = np.sqrt(2 * np.pi / finesse)
        gamma = t_in**2 / 2
        omega_cav = gamma / tau
        linewidth = c / (2 * arm_length * finesse)
        signal_recycling_reflection = np.sqrt(1 - signal_recycling_transmission**2)
    
        chi = 1 / ( m_eff * (omega_m**2 - omega**2 - i * (omega_m * omega / mech_quality_factor) ) )

        # diag = (gamma + (i * omega * tau)) / (gamma - (i * omega * tau))
        K = 32 * input_intensity * h_bar * k**2 / ((1 + omega**2 / omega_cav**2) * gamma**2) * chi
        
        Phi = omega * L_recycling / c + np.arctan(2 * omega / linewidth)
        K_sr = K * signal_recycling_transmission**2 / abs(1 + np.exp(2 * i * Phi) * signal_recycling_reflection)**2
        
        # print(abs((K_sr - K_sr1) / K_sr))
        
        LinearOpticalElement.__init__(self, np.array([[1, 0], [K_sr, 1]]))


class FilterCavity(LinearOpticalElement):
    '''Filter cavity in the perfectly mode-matched case'''
    
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
    
    def reflectionCoefficient(self):
        '''gives the transfer function in field amplitude of the cavity'''
        
        epsilon = 2 * self.losses / (self.input_transmission**2 + self.losses)
        ksi = 4 * (self.omega - self.detuning) * self.length / (c * (self.input_transmission**2 + self.losses))
        
        return( 1 - (2 - epsilon) / (1 + i * ksi) )

        

class ModeMismatchedFilterCavity(OpticalElement):
    '''Filter cavity including the losses due to squeezer-cavity and local-oscillator-cavity mode-mismatches
    In case the mode-mismatch is negligible, use FilterCavity'''
    
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
            cav_losses = 1 - (abs(t00 * FilterCavity(omega, detuning, length, input_transmission, losses).reflectionCoefficient() + tmm)**2 + abs(t00 * FilterCavity(-omega, detuning, length, input_transmission, losses).reflectionCoefficient() + tmm)**2) / 2
            cav_vac = cav_losses * State().covariance_matrix
                        
            return cav + cav_vac
        
        OpticalElement.__init__(self, transfer)



