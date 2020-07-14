# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:06:37 2020

@author: pielsticker
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

class Spectrum:
    def __init__(self,start,stop,step,label):
        self.start = start
        self.stop = stop
        self.step = step
        self.x = np.flip(np.arange(
                        self.start,self.stop+self.step,self.step))
        self.clear_lineshape()
        self.label = label
        
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
        filename = filename
        self.label, self.data = self.convert(filename)
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
    
    def convert(self, filename):
        file = open(filename,'r')
        lines = []
        for line in file.readlines():
            lines += [line]
        # This takes the species given in the first line
        label = str(lines[0]).split('\n')[0] 
        lines = lines[1:]
        lines = [[float(i) for i in line.split()] for line in lines]
        data = np.array(lines)
        
        return label, data

#%%

def convert_spectra(filenames, plot_all = True):
    datafolder = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\data\measured'
    X = np.zeros((len(filenames),1121,1))
    y = np.zeros((len(filenames),4))

    for name in filenames:
        filepath = os.path.join(datafolder, name)
        spectrum = MeasuredSpectrum(filepath)
        spectrum.start = 694
        spectrum.stop = 750
        spectrum.step = 0.05
        spectrum.update_range()
    
        if spectrum.lineshape.shape[0] != 1121:
            old_lineshape = spectrum.lineshape
            new_lineshape = np.zeros((1121,))  
        
            for point in old_lineshape:
                index = np.where(old_lineshape == point)[0]
                try:
                    new = old_lineshape[index+1][0]
                except:
                    new = old_lineshape[index][0]
                mean = np.mean(np.array([point,new]))
                new_lineshape[index*2] = point
                try:
                    new_lineshape[index*2+1] = mean
                except:
                    pass
    
                diff_len = new_lineshape.shape[0] - old_lineshape.shape[0]*2
                end_value = old_lineshape[-1]
                new_lineshape[new_lineshape.shape[0]-diff_len:] = end_value
    
                spectrum.lineshape = new_lineshape
    
        spectrum.normalize()
        index = filenames.index(name)
        X[index] = np.reshape(spectrum.lineshape, (-1,1))
        label = [float(i) for i in spectrum.label.split()]
        y[index] = np.reshape(np.array(label), (1,-1))
        
        if plot_all:
            plt.plot(spectrum.x,spectrum.lineshape)
            plt.xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
            plt.legend([label])
            plt.tight_layout()
            plt.show()
    
    return X, y

#%%
filenames = ['Fe_metal_Mark.txt',
             'FeO_Mark.txt',
             'Fe3O4_Mark.txt',
             'Fe2O3_Mark.txt',
             'test.txt']

output_file= r'C:\Users\pielsticker\Simulations\measured.h5py'
    
with h5py.File(output_file, 'w') as hf:
    X, y = convert_spectra(filenames, plot_all = True)
    hf.create_dataset('X', data = X,
                      compression="gzip", chunks=True,
                      maxshape=(None,X.shape[1],X.shape[2]))        
    hf.create_dataset('y', data = y,
                      compression="gzip", chunks=True,
                      maxshape=(None, y.shape[1]))
    
#%% Check   
with h5py.File(output_file, 'r') as hf:
    dataset_size = hf['X'].shape[0]
    X_load = hf['X'][:,:,:]
    y_load = hf['y'][:,:]
    print(np.allclose(X,X_load))
    print(np.allclose(y,y_load))

