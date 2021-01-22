# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:55:11 2021

@author: Alexandre
"""

import numpy as np
from simulation_library.constants import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons


def noise_spectrum(setup, omega_array, detuning, input_transmission, phase_mm_default, input_state = sm.State(), dB = False, logscale = True, sliders = False, multiple_phases = False):
    
    nb_freq = len(omega_array)
    freq_min = omega_array[0]
    freq_max = omega_array[-1]
    
    angle_init = 90 # °
    angle_step = 1
    detuning_init = detuning * 1e-6 #s-1
    detuning_step = 1e-2
    detuning_min = 1.1
    detuning_max = 1.7
    transmission_init = input_transmission
    transmission_min = 0
    transmission_max = 0.1
    transmission_step = 0.001
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
#    ax.set_ylim(-1, 10)

    if (not dB) and logscale:
        plt.yscale('log')
    
    if multiple_phases:
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
                var_plot[j] = covariance_matrix.variance(0, dB)
                
                variance_dB(cov_ref_ro(omega[j], 2 * np.pi * detuning_init * 1e6, transmission_init, phase_mm[k]), (np.pi/180) * angle_init)
                
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