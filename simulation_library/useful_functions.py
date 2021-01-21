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