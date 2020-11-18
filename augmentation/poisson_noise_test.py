# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:15:57 2020

@author: Mark
"""
from numpy.random import poisson
import matplotlib.pyplot as plt


#%%
l = 10000
a = poisson(l,100) / l

plt.plot(a)
