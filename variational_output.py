# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:35:05 2021

@author: Alexandre

SQZ -> IFO -> FC -> RO
"""

#%% Importations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
import cmath as cm

#%% Choix du graphe plotté

phases_multiples = False
sliders = False
dB = False
logscale = True

#%% Constants

c = 299792458 #m.s-1
i = complex(0, 1)
h_bar = 1.054571e-34 #J.s

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
injection_losses = 0 # à modifier -> pertes avant la fc
propagation_losses = 0 # à modifier -> pertes après la fc, avant l'ifo
readout_losses = 0 # à modifier -> pertes après l'ifo
mode_mismatch_squeezer_filter_cavity = 0.0
mode_mismatch_squeezer_local_oscillator = 0.0

# Squeezing factor and squeezed quadrature
squeezing_dB = 5
squeezing_angle = np.pi/2 # dB, rad

phase_mm_default = np.pi  # worst-case scenario ?

# Windows of frequencies observed
freq_span = 1e6 #Hz

freq_min = omega_m / (2 * np.pi) - freq_span
freq_max = omega_m / (2 * np.pi) + freq_span

omega_min = 2 * np.pi * freq_min #s-1
omega_max = 2 * np.pi * freq_max #s-1 <- range of sideband frequencies observed
nb_freq = 1000 # <- freq resolution

omega = np.linspace(omega_min, omega_max, nb_freq)



#%% Useful computations

#def reflection(omega, detuning):
#    '''gives the transfer function in field amplitude of the cavity'''
#        
#    return( ( r1 - r2*(r1**2 + t1**2)*np.exp(2 * i * (omega - detuning) * L / c) ) / (1 - r1*r2*np.exp(2 * i * (omega - detuning) * L / c)))

vacuum_cov = np.eye(2)


def reflection(omega, detuning, input_transmission):
    '''gives the transfer function in field amplitude of the cavity'''
    
    epsilon = 2 * filter_cavity_losses / (input_transmission**2 + filter_cavity_losses)
    ksi = 4 * (omega - detuning) * L_fc / (c * (input_transmission**2 + filter_cavity_losses))
    
    return( 1 - (2 - epsilon) / (1 + i * ksi) )

def rotation(theta):
    '''returns the 2x2 rotation matrix of angle theta'''
    
    return np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

A2 = (1 / np.sqrt(2)) * np.matrix([[1, 1], [-i, i]])

def apply(mat, cov):
    '''returns the new covariance matrix after passage through an optical component with two-photon matrix mat'''

    return mat.dot(cov).dot(np.transpose(np.conjugate(mat)))

#%% Computation of the variance of any quadrature
    
def variance(cov, theta):
    '''gives the variance of the quadrature X_theta, given a covariance matrix cov'''
    
    cov_rot = apply(rotation(-theta), cov)

    return np.real(cov_rot[0, 0])

def variance_dB(cov, theta):
    '''gives the variance of the quadrature X_theta in dB wrt the vacuum noise, given a covariance matrix cov'''
    
    return  10 * np.log10(variance(cov, theta))

def Sxx(cov, theta, omega):
    '''gives the measurement noise in m^2/Hz'''
    
    return lambda_carrier / (256 * finesse**2 * intensity_input) * variance(cov, theta) * (1 + (omega/omega_cav)**2) 

#%% Transfer matrices of the optical components

def SQZ(sqz_dB, theta):
    '''two-photon matrix of the squeezer'''
    
    r = - 0.5 * np.log(10**(-sqz_dB/10))
    amplitude_squeezed = np.matrix( [[np.exp(-r), 0], [0, np.exp(r)]] )
    
    return apply(rotation(-theta), amplitude_squeezed)

def FC(omega, detuning, input_transmission):
    '''two-photon matrix of the filter cavity'''
    
    one_photon_matrix = np.matrix([[reflection(omega, detuning, input_transmission), 0], [0, np.conjugate(reflection(-omega, detuning, input_transmission))]])
    
    return apply(A2, one_photon_matrix)

def IFO(omega):
    '''two-photon matrix of the interferometer (here, a cavity with a vibrating membrane)'''
    
    k = 2 * np.pi / lambda_carrier
    
    chi = 1 / ( m_eff * (omega_m**2 - omega**2 - i * (omega_m * omega / Q) ) )
#    print((omega_m**2 - omega**2 - i * (omega_m * omega / Q)).real / (omega_m**2 - omega**2 - i * (omega_m * omega / Q)).imag)
    diag = (gamma + (i * omega * tau)) / (gamma + (i * omega * tau))
    K = ( 16 * t_in * intensity_input * h_bar * k**2 / (gamma - i*omega*tau)**2 ) * chi
    
    return np.matrix([[diag, 0], [K, diag]])

#%% Application on the covariance matrix for a squeezed field
    
cov_init = vacuum_cov # the initial covariance matrix is the identity (vaccuum)

cov_squeezed = apply(SQZ(squeezing_dB, squeezing_angle), cov_init) # squeezed vacuum 

cov_squeezed_inj_loss = injection_losses * vacuum_cov + (1 - injection_losses) * cov_squeezed # squeezed vacuum degraded by injection losses

def cov_ifo(omega):
    '''covariance matrix after the interferometer'''
    
    return apply(IFO(omega), cov_squeezed_inj_loss)

def cov_ref(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the filter cavity'''
    
    # mode mismatch
    a0 = np.sqrt(1 - mode_mismatch_squeezer_filter_cavity)
    c0 = np.sqrt(1 - mode_mismatch_squeezer_local_oscillator)
    b0 = a0 * c0 + np.exp(i * phase_mm) * np.sqrt((1 - a0**2) * (1 - c0**2))
    t00 = a0 * np.conjugate(b0)
    tmm = c0 - t00
        
    MM = abs(tmm) * rotation(cm.phase(tmm))
    
    cav = apply(t00 * FC(omega, detuning, input_transmission) + MM, cov_ifo(omega))
    cav_losses = 1 - (abs(t00 * reflection(omega, detuning, input_transmission) + tmm)**2 + abs(t00 * reflection(-omega, detuning, input_transmission) + tmm)**2) / 2
    cav_vac = cav_losses * vacuum_cov
    
    return cav + cav_vac

def cov_ref_lossy(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the cavity degraded by propagation losses'''
    
    return propagation_losses * vacuum_cov + (1 - propagation_losses) * cov_ref(omega, detuning, input_transmission, phase_mm)

def cov_ref_ro(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the interferometer degraded by the reandout losses'''
    
    return readout_losses * vacuum_cov + (1 - readout_losses) * cov_ref_lossy(omega, detuning, input_transmission, phase_mm)

#%% Same without squeezing
    
cov_init_vac = vacuum_cov # the initial covariance matrix is the identity (vaccuum)

cov_squeezed_inj_loss_vac = injection_losses * vacuum_cov + (1 - injection_losses) * cov_init_vac # squeezed vacuum degraded by injection losses

def cov_ifo_vac(omega):
    '''covariance matrix after the interferometer'''
    
    return apply(IFO(omega), cov_squeezed_inj_loss_vac)

def cov_ref_vac(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the filter cavity'''
    
    # mode mismatch
    a0 = np.sqrt(1 - mode_mismatch_squeezer_filter_cavity)
    c0 = np.sqrt(1 - mode_mismatch_squeezer_local_oscillator)
    b0 = a0 * c0 + np.exp(i * phase_mm) * np.sqrt((1 - a0**2) * (1 - c0**2))
    t00 = a0 * np.conjugate(b0)
    tmm = c0 - t00
        
    MM = abs(tmm) * rotation(cm.phase(tmm))
    
    cav = apply(t00 * FC(omega, detuning, input_transmission) + MM, cov_ifo_vac(omega))
    cav_losses = 1 - (abs(t00 * reflection(omega, detuning, input_transmission) + tmm)**2 + abs(t00 * reflection(-omega, detuning, input_transmission) + tmm)**2) / 2
    cav_vac = cav_losses * vacuum_cov
    
    return cav + cav_vac

def cov_ref_lossy_vac(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the cavity degraded by propagation losses'''
    
    return propagation_losses * vacuum_cov + (1 - propagation_losses) * cov_ref_vac(omega, detuning, input_transmission, phase_mm)

def cov_ref_ro_vac(omega, detuning, input_transmission, phase_mm):
    '''covariance matrix after the interferometer degraded by the reandout losses'''
    
    return readout_losses * vacuum_cov + (1 - readout_losses) * cov_ref_lossy_vac(omega, detuning, input_transmission, phase_mm)



#%% Graphical interface

if __name__ == '__main__':
    
    angle_init = 90 # °
    angle_step = 1
    detuning_init = detuning * 1e-6 #s-1
    detuning_step = 1e-2
    detuning_min = 1.1
    detuning_max = 1.7
    transmission_init = t1
    transmission_min = 0
    transmission_max = 0.1
    transmission_step = 0.001
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
#    ax.set_ylim(-1, 10)

    if (not dB) and logscale:
        plt.yscale('log')
    
    if phases_multiples:
        phase_mm = 2 * np.pi * np.arange(0, 8) / 8
    else:
        phase_mm = [phase_mm_default]
    l = list()
    v = list()
        
    for k in range(len(phase_mm)):
        var_plot = np.zeros(nb_freq)
        var_plot_vac = np.zeros(nb_freq)
        for j in range(nb_freq): # j indexes frequencies
            if dB:
                var_plot[j] = variance_dB(cov_ref_ro(omega[j], 2 * np.pi * detuning_init * 1e6, transmission_init, phase_mm[k]), (np.pi/180) * angle_init)
                var_plot_vac[j] = variance_dB(cov_ref_ro_vac(omega[j], 2 * np.pi * detuning_init * 1e6, transmission_init, phase_mm[k]), (np.pi/180) * angle_init)
            else:
                var_plot[j] = Sxx(cov_ref_ro(omega[j], 2 * np.pi * detuning_init * 1e6, transmission_init, phase_mm[k]), (np.pi/180) * angle_init, omega[j])
                var_plot_vac[j] = Sxx(cov_ref_ro_vac(omega[j], 2 * np.pi * detuning_init * 1e6, transmission_init, phase_mm[k]), (np.pi/180) * angle_init, omega[j])
        l.append(0)
        v.append(0)
        l[k] = ax.plot(omega/(2*np.pi), var_plot, '-') #, label='Mismatch phase = {}°'.format(phase_mm[k] * 180 / np.pi)
#        plt.legend()
        v[k] = ax.plot(omega/(2*np.pi), var_plot_vac, '--', color='black', linewidth=1)
        
    if sliders:
        
        axcolor = 'lightgoldenrodyellow'
        axangle = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axdetuning = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axtransmission = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        axfreqmin = plt.axes([0.25, 0.25, 0.1, 0.05], facecolor=axcolor)
        axfreqmax = plt.axes([0.6, 0.25, 0.1, 0.05], facecolor=axcolor)
        
        sangle = Slider(axangle, 'Homodyne angle (°)', 0, 180, valinit=angle_init, valstep=angle_step)
        sdetuning = Slider(axdetuning, 'Detuning (MHz)', detuning_min, detuning_max, valinit=detuning_init, valstep=detuning_step)
        stransmission = Slider(axtransmission, 'Input mirror transmission', transmission_min, transmission_max, valinit=transmission_init, valstep=transmission_step)
        sfreqmin = TextBox(axfreqmin, 'Min frequency (MHz)', initial=str(freq_min * 1e-6), color='.95', hovercolor='1')
        sfreqmax = TextBox(axfreqmax, 'Max frequency (MHz)', initial=str(freq_max * 1e-6), color='.95', hovercolor='1')
        
        def update(val):
            angle = np.pi * sangle.val /180
            detuning = 2 * np.pi * sdetuning.val * 1e6
            transmission = stransmission.val
            omega_min = 2 * np.pi * float(sfreqmin.text) * 1e6 #s-1
            omega_max = 2 * np.pi * float(sfreqmax.text) * 1e6 #s-1 <- range of sideband frequencies observed
    #        nb_freq = 1000 # <- freq resolution
            omega = np.linspace(omega_min, omega_max, nb_freq)
            for k in range(len(phase_mm)):
                var_plot = np.zeros(nb_freq)
                var_plot_vac = np.zeros(nb_freq)
                for j in range(nb_freq):
                    if dB:
                        var_plot[j] = variance_dB(cov_ref_ro(omega[j], detuning, transmission, phase_mm[k]), angle)
                        var_plot_vac[j] = variance_dB(cov_ref_ro_vac(omega[j], detuning, transmission, phase_mm[k]), angle)
                    else:
                        var_plot[j] = Sxx(cov_ref_ro(omega[j], detuning, transmission, phase_mm[k]), angle, omega[j])
                        var_plot_vac[j] = Sxx(cov_ref_ro_vac(omega[j], detuning, transmission, phase_mm[k]), angle, omega[j])
                l[k][0].set_ydata(var_plot)    
                v[k][0].set_ydata(var_plot_vac)
            fig.canvas.draw()
        
        sangle.on_changed(update)
        sdetuning.on_changed(update)
        stransmission.on_changed(update)
        sfreqmin.on_submit(update)
        sfreqmax.on_submit(update)
    
    else:
        
        axcolor = 'lightgoldenrodyellow'
        axangle = plt.axes([0.25, 0.1, 0.1, 0.05], facecolor=axcolor)
        axdetuning = plt.axes([0.6, 0.175, 0.1, 0.05], facecolor=axcolor)
        axtransmission = plt.axes([0.25, 0.175, 0.1, 0.05], facecolor=axcolor)
        axfreqmin = plt.axes([0.25, 0.25, 0.1, 0.05], facecolor=axcolor)
        axfreqmax = plt.axes([0.6, 0.25, 0.1, 0.05], facecolor=axcolor)

        sangle = TextBox(axangle, 'Homodyne angle (°)', initial=str(angle_init), color='.95', hovercolor='1')
        sdetuning = TextBox(axdetuning, 'Detuning (MHz)', initial=str(detuning_init), color='.95', hovercolor='1')
        stransmission = TextBox(axtransmission, 'Input mirror transmission', initial=str(transmission_init), color='.95', hovercolor='1')
        sfreqmin = TextBox(axfreqmin, 'Min frequency (MHz)', initial=str(freq_min * 1e-6), color='.95', hovercolor='1')
        sfreqmax = TextBox(axfreqmax, 'Max frequency (MHz)', initial=str(freq_max * 1e-6), color='.95', hovercolor='1')
        
        def update(val):
            angle = np.pi * float(sangle.text) /180
            detuning = 2 * np.pi * float(sdetuning.text) * 1e6
            transmission = float(stransmission.text)
            omega_min = 2 * np.pi * float(sfreqmin.text) * 1e6 #s-1
            omega_max = 2 * np.pi * float(sfreqmax.text) * 1e6 #s-1 <- range of sideband frequencies observed
    #        nb_freq = 1000 # <- freq resolution
            omega = np.linspace(omega_min, omega_max, nb_freq)
            for k in range(len(phase_mm)):
                var_plot = np.zeros(nb_freq)
                var_plot_vac = np.zeros(nb_freq)
                for j in range(nb_freq):
                    if dB:
                        var_plot[j] = variance_dB(cov_ref_ro(omega[j], detuning, transmission, phase_mm[k]), angle)
                        var_plot_vac[j] = variance_dB(cov_ref_ro_vac(omega[j], detuning, transmission, phase_mm[k]), angle)
                    else:
                        var_plot[j] = Sxx(cov_ref_ro(omega[j], detuning, transmission, phase_mm[k]), angle, omega[j])
                        var_plot_vac[j] = Sxx(cov_ref_ro_vac(omega[j], detuning, transmission, phase_mm[k]), angle, omega[j])
                l[k][0].set_ydata(var_plot)
                v[k][0].set_ydata(var_plot_vac)
            fig.canvas.draw()
        
        sangle.on_submit(update)
        sdetuning.on_submit(update)
        stransmission.on_submit(update)
        sfreqmin.on_submit(update)
        sfreqmax.on_submit(update)    
    
    plt.show()
