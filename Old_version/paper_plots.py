# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:15:59 2020

@author: alexa
"""

#%% Importations

import numpy as np
import matplotlib.pyplot as plt
import cmath as cm

#%% Choix du graphe plotté

graphe = 1
phases_multiples = False

#%% Constants

c = 299792458 #m.s-1
i = complex(0, 1)
h_bar = 1.054571e-34 #J.s

#%% Parameters definition -> Schnabel 2005

if graphe == 1 or graphe == 2:
    
    L_fc = 50e-2 #m
    detuning = - 2 * np.pi * 15.15e6 #s-1
    r1, r2 = np.sqrt(0.97), np.sqrt(0.9995)
    t1, t2 = np.sqrt(1 - r1**2), np.sqrt(1 - r2**2)
    
    #Frequency window observed    /!\ omega = detuning from the carrier (sidebands)
    omega_min = 2 * np.pi * 12e6 #s-1
    omega_max = 2 * np.pi * 18e6 #s-1 <- range of frequency sidebands observes
    nb_freq = 1000 # <- freq resolution
    
    omega = np.linspace(omega_min, omega_max, nb_freq)
    
    squeezing_dB, squeezing_angle = 2, 0
    
    filter_cavity_losses = 0
    propagation_losses = 0
    mode_mismatch_squeezer_filter_cavity = 0
    mode_mismatch_squeezer_local_oscillator = 0
    filter_cavity_length_noise = 0 # m -> Not used here 
    
    injection_losses = 0 # à modifier -> pertes avant la fc
    propagation_losses = 0 # à modifier -> pertes après la fc, avant l'ifo
    readout_losses = 0 # à modifier -> pertes après l'ifo
    
    phase_mm_default = 0

#%% Parameters definition -> Jap paper 2020

if graphe == 3:
    
    intensity_input = 5e23
    lambda_carrier = 1064e-9 #m
    
    # filter cavity
    L_fc = 300 #m
    t1, t2 = np.sqrt(0.00136), np.sqrt(3.9e-6)
    
    # interferometer
    #L_ifo = 300
    #t_in = 0.9
    #m_eff = 1e-6 #kg
    #omega_m = 2 * np.pi * 500#Hz
    #Q = 100
    
    # Paper parameters
    filter_cavity_losses = 120e-6
    propagation_losses = 0.36
    mode_mismatch_squeezer_filter_cavity = 0.06
    mode_mismatch_squeezer_local_oscillator = 0.02
    filter_cavity_length_noise = 6e-12 # m -> Not used here 
    phase_noise = 30e-3 # rad -> Not used here
    squeezing_dB, squeezing_angle = 8.3, 0 # dB, rad
    
    injection_losses = 0 # à modifier -> pertes avant la fc
    propagation_losses = 0.36 # à modifier -> pertes après la fc, avant l'ifo
    readout_losses = 0 # à modifier -> pertes après l'ifo
    
    phase_mm_default = np.pi  # à trouver pour le pire des cas
    
    omega_min = 2 * np.pi * 40 #s-1
    omega_max = 2 * np.pi * 1000 #s-1 <- range of sideband frequencies observed
    nb_freq = 1000 # <- freq resolution
    
    omega = np.linspace(omega_min, omega_max, nb_freq)



#%% Useful computations

#def reflection(omega, detuning):
#    '''gives the transfer function in field amplitude of the cavity'''
#        
#    return( ( r1 - r2*(r1**2 + t1**2)*np.exp(2 * i * (omega - detuning) * L / c) ) / (1 - r1*r2*np.exp(2 * i * (omega - detuning) * L / c)))

def reflection(omega, detuning, input_transmission):
    '''gives the transfer function in field amplitude of the cavity'''
    
    epsilon = 2 * filter_cavity_losses / (input_transmission**2 + filter_cavity_losses)
    ksi = 4 * (omega - detuning) * L_fc / (c * (input_transmission**2 + filter_cavity_losses))
    
    return( 1 - (2 - epsilon) / (1 + i * ksi) )

def rotation(theta):
    
    return np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

A2 = (1 / np.sqrt(2)) * np.matrix([[1, 1], [-i, i]])

def apply(mat, cov):

    return mat.dot(cov).dot(np.transpose(np.conjugate(mat)))

#%% Transfer matrices of the optical components

def SQZ(sqz_dB, theta):
    
    r = - 0.5 * np.log(10**(-sqz_dB/10))
    amplitude_squeezed = np.matrix( [[np.exp(-r), 0], [0, np.exp(r)]] )
    
    return apply(rotation(-theta), amplitude_squeezed)

def FC(omega, detuning, input_transmission):
    
    one_photon_matrix = np.matrix([[reflection(omega, detuning, input_transmission), 0], [0, np.conjugate(reflection(-omega, detuning, input_transmission))]])
    
    return apply(A2, one_photon_matrix)

def IFO(omega):
    
#    chi = 1 / ( m_eff * (omega_m**2 - omega**2 - i * omega_m * omega / Q ) )
#    diag = ((t_in**2 / 2) + (i * omega * 2 * L_ifo / c)) / ((t_in**2 / 2) + (i * omega * 2 * L_ifo / c))
#    K = 16 * t_in * intensity_input * h_bar * 2 * np.pi / (lambda_carrier * ((t_in**2/2) - i*omega*2*L_ifo/c)**2) * chi
#    one_photon_matrix = np.matrix([[diag, 0], [-K, diag]])
#    
#    return apply(A2, one_photon_matrix)
    return np.eye(2)

#%% Application on the covariance matrix
    
cov_init = np.eye(2)

cov_squeezed = apply(SQZ(squeezing_dB, squeezing_angle), cov_init)

cov_squeezed_inj_loss = injection_losses * np.eye(2) + (1 - injection_losses) * cov_squeezed

def cov_ref(omega, detuning, input_transmission, phase_mm):
    
    # mode mismatch
    a0 = np.sqrt(1 - mode_mismatch_squeezer_filter_cavity)
    c0 = np.sqrt(1 - mode_mismatch_squeezer_local_oscillator)
    b0 = a0 * c0 + np.exp(i * phase_mm) * np.sqrt((1 - a0**2) * (1 - c0**2))
    t00 = a0 * np.conjugate(b0)
    tmm = c0 - t00
        
    MM = abs(tmm) * rotation(cm.phase(tmm))
    
    cav = apply(t00 * FC(omega, detuning, input_transmission) + MM, cov_squeezed_inj_loss)
    cav_losses = 1 - (abs(t00 * reflection(omega, detuning, input_transmission) + tmm)**2 + abs(t00 * reflection(-omega, detuning, input_transmission) + tmm)**2) / 2
    cav_vac = cav_losses * np.eye(2)
    
    return cav + cav_vac

def cov_ref_lossy(omega, detuning, input_transmission, phase_mm):
    
    return propagation_losses * np.eye(2) + (1 - propagation_losses) * cov_ref(omega, detuning, input_transmission, phase_mm)

def cov_ref_ifo(omega, detuning, input_transmission, phase_mm):
    
    return apply(IFO(omega), cov_ref_lossy(omega, detuning, input_transmission, phase_mm))

def cov_ref_ro(omega, detuning, input_transmission, phase_mm):
    
    return readout_losses * np.eye(2) + (1 - readout_losses) * cov_ref(omega, detuning, input_transmission, phase_mm)


#%% Computation of the variance of any quadrature
    
def variance(cov, theta):
    
    cov_rot = apply(rotation(-theta), cov)

    return np.real(cov_rot[0, 0])

def variance_dB(cov, theta):
    
    return  10 * np.log10(variance(cov, theta))

#%% Generation of the Wigner function from the covariance matrix

def pdf(cov, resolution):
    
    wigner = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            xy = np.array([10 * 2/resolution * (i - resolution/2),10 * 2/resolution * (j - resolution/2)])
            wigner[i,j] = np.exp(-0.5 * np.transpose(xy).dot(cov).dot(xy))
    
    return wigner

#%% Wigner functions at various frequencies (Schnabel)

if graphe == 1:
    
    if __name__ == '__main__':
        
        freq = np.array([12, 13, 13.8, 14.1, 14.4, 14.7, 15.3, 15.6, 15.9, 16.2, 17, 18]) * (1e6)
        subplt = 1
        
        for f in freq:
            plt.subplot(4, 3, subplt)
            plt.axis('off')
            plt.title('{} MHz'.format(round(f / (1e5)) / 10))
            plt.imshow(pdf(np.real(cov_ref_ro(2*np.pi*f, detuning, t1, phase_mm_default)), 100))
            subplt += 1

#%% Variance vs frequency (Schnabel)

if graphe == 2:
    
    if __name__ == '__main__':
      
        for k in range(10):
                
            theta_quad = k * 10 * np.pi / 180 #angle of the quadrature one wishes to measure
            
            var_plot = np.zeros(nb_freq)
            for j in range(nb_freq):
                var_plot[j] = variance_dB(cov_ref_ifo(omega[j], detuning, t1, phase_mm_default), theta_quad)
            plt.plot(omega/(2*np.pi), var_plot, label='{}°'.format(k*10))
            plt.legend()

#%% Variance vs frequency (Zhao et al)

if graphe == 3:
    
    if __name__ == '__main__':
      
        angle = np.array([-2.2, 15.5, 27.7, 39.4, 60.1, 92.8]) # degrees
        detuning = np.array([42.6, 69.2, 62.2, 60.4, 67.9, 71.4]) # Hz
        
        for k in range(len(angle)):
                
            theta_quad = angle[k] * np.pi / 180 #angle of the quadrature one wishes to measure
            detuning_cav = 2 * np.pi * detuning[k]
            
            plt.xscale('log')
            var_plot = np.zeros(nb_freq)
            for j in range(nb_freq):
                var_plot[j] = variance_dB(cov_ref_ifo(omega[j], detuning_cav, t1, phase_mm_default), theta_quad) #minus sign because theta defined as theta_squeezed_field - theta_LO
            plt.plot(omega/(2*np.pi), var_plot, label='Angle = {}°, detuning = {} Hz'.format(angle[k], detuning[k]))
            plt.legend()

