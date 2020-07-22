# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:06:37 2020

@author: pielsticker
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class Spectrum:
    def __init__(self,start,stop,step,label):
        self.start = start
        self.stop = stop
        self.step = step
        self.x = np.flip(np.arange(
                        self.start,self.stop+self.step,self.step))
        self.clear_lineshape()
        self.label = label
        self.type = None
        
    def clear_lineshape(self):
        self.lineshape = np.zeros(
            int(np.round((self.stop-self.start+self.step)/self.step,
                         decimals = 1)))
        self.x = np.flip(np.arange(self.start,self.stop+self.step,self.step))


    def normalize(self):
        if np.sum(self.lineshape) != 0:
            self.lineshape = self.lineshape / np.nansum(self.lineshape) 
    
    def update_range(self):
        self.x = np.flip(np.arange(self.start,self.stop+self.step,self.step))
            
class MeasuredSpectrum(Spectrum):
    def __init__(self, filename):
        self.type = 'measured'
        self.filename = filename
        self.label, self.data = self.convert(self.filename)
        x = self.data[:,0]
        
        # Determine the step size
        x1 = np.roll(x,-1)
        diff = np.abs(np.subtract(x,x1))
        self.step = round(np.min(diff[diff!=0]),2)
        x = x[diff !=0]
        self.start = np.min(x)
        self.stop = np.max(x)
        Spectrum.__init__(self, self.start,self.stop,self.step, self.label)
        self.x = x
        self.lineshape = self.data[:,1][diff != 0]
        #self.normalize()
    
    def convert(self, filename):
        with open(filename,'r') as file:
            lines = []
            for line in file.readlines():
                lines += [line]
            # This takes the species given in the first line
            self.species = str(lines[0]).split('\n')[0] 
            lines = lines[1:]
            lines = [[float(i) for i in line.split()] for line in lines]
            data = np.array(lines)
            # The label is a dictionary of the form
            # {species: concentration}.
            label = {self.species: 1.0}
        
        return label, data

#%%
datafolder = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\data\references'
filenames = ['Fe_metal_Mark_shifted.txt','FeO_Mark_shifted.txt',
             'Fe3O4_Mark_shifted.txt','Fe2O3_Mark_shifted.txt']

legend_list = []
for filename in filenames:
    filepath = os.path.join(datafolder, filename)
    spectrum = MeasuredSpectrum(filepath)
    spectrum.start = 694
    spectrum.stop = 750
    spectrum.step = 0.05
    spectrum.update_range()
    #spectrum.normalize()
    label = spectrum.label
    old_lineshape = spectrum.lineshape
    new_lineshape = np.zeros((1121,))
    
    for point in old_lineshape:
        index = np.where(old_lineshape == point)[0]
        try:
            new = old_lineshape[index+1][0]
        except:
            new = old_lineshape[index][0]
        a = np.array([point,new])
        mean = np.mean(a)
        new_lineshape[index*2] = point
        new_lineshape[index*2+1] = mean
    
    diff_len = new_lineshape.shape[0] - old_lineshape.shape[0]*2
    end_value = old_lineshape[-1]
    new_lineshape[new_lineshape.shape[0]-diff_len:] = end_value
    
    filename_new = filename.split('.')[0] + '_new.txt'
    filepath_new = os.path.join(datafolder, filename_new)
    with open(filepath_new,'w') as file:
        lines = [spectrum.species + '\n']
        for i in range(len(spectrum.x)):
            lines.append(str('{:e}'.format(spectrum.x[i])) + ' ' + str('{:e}'.format(new_lineshape[i]))+ '\n')
        file.writelines(lines)
        
    plt.plot(np.flip(spectrum.x),new_lineshape)
    legend_list.append(label)
plt.legend(legend_list)
                
#%% 
datafolder = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\data\references'
filenames = ['Fe_metal_Mark_shifted_new.txt', 'FeO_Mark_shifted_new.txt',
             'Fe3O4_Mark_shifted_new.txt', 'Fe2O3_Mark_shifted_new.txt']
legend_list = []
spectra = []
for filename in filenames:
    filepath = os.path.join(datafolder, filename)
    spectrum = MeasuredSpectrum(filepath)
    plt.plot(spectrum.x,spectrum.lineshape)
    label = spectrum.label
    legend_list.append(label)
    spectra.append(spectrum)
plt.legend(legend_list) 

#%% 
energies = []
for i in range(4):
    spectrum = spectra[i]
    arg_max = np.argmax(spectrum.lineshape, axis = 0)
    energies.append(spectrum.x[arg_max])
    print(arg_max, spectrum.label, energies[i])

