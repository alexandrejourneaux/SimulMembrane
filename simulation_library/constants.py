# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:43:19 2021

@author: Alexandre
"""

import numpy as np

c = 299792458 #m.s-1
i = complex(0, 1)
h_bar = 1.054571e-34 #J.s

# Change of frame matrix from one-photon to two-photon formalism
A2 = (1 / np.sqrt(2)) * np.matrix([[1, 1], [-i, i]])