# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:31:01 2021

@author: Alexandre
"""

import numpy as np

def rotation(theta):
    '''returns the 2x2 rotation matrix of angle theta'''
    
    return np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def apply(mat, cov):
    '''returns the new covariance matrix after passage through an optical component with two-photon matrix mat'''

    return mat.dot(cov).dot(np.transpose(np.conjugate(mat)))

def pdf(cov, resolution):
    
    wigner = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            xy = np.array([10 * 2/resolution * (i - resolution/2),10 * 2/resolution * (j - resolution/2)])
            wigner[i,j] = np.exp(-0.5 * np.transpose(xy).dot(cov).dot(xy))
    
    return wigner