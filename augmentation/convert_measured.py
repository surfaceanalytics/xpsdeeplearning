# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:06:37 2020

@author: pielsticker
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd

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
        self.label, self.data, self.number = self.convert(filename)
        x = self.data[:,0]
        
        # Determine the step size
        x1 = np.roll(x,-1)
        x2 = np.roll(x,-2)
        diff = np.abs(np.subtract(x,x1))
        self.step = round(np.min(diff[diff!=0]),3)
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
        label = str(lines[0]).split(' ', maxsplit = 2)[2].split(':')[0]
        number = int(str(lines[0]).split(' ', maxsplit = 2)[1].split(':')[1])
        lines = lines[8:]
        lines = [[float(i) for i in line.split()] for line in lines]
        data = np.array(lines)[:,2:]
        
        return label, data, number

#%%
def get_labels():
    filepath = r'C:\Users\pielsticker\Desktop\Mixed iron spectra\peak fits.xlsx'
    df = pd.read_excel(filepath)
    
    y = np.transpose(np.array([df['Fe metal'],df['FeO'],df['Fe3O4'],df['Fe2O3']]))
    names = np.reshape(np.array(list(df['name'])),(-1,1))
                        
    return y, names

def convert_spectra(plot_all = True):
    input_datafolder = r'C:\Users\pielsticker\Desktop\Mixed iron spectra\exported'
    
    filenames = next(os.walk(input_datafolder))[2]
    
    X = np.zeros((len(filenames),1121,1))
    y, names = get_labels()
    
    spectra = []
    
    for name in filenames:
        filepath = os.path.join(input_datafolder, name)
        spectrum = MeasuredSpectrum(filepath)
        spectrum.start = 694
        spectrum.stop = 750
        spectrum.step = 0.05
        
        x = list(spectrum.x)
        try:  
            start = x.index(np.float64(750))
            end = x.index(np.float64(694))+1
            
            spectrum.data = spectrum.data[start:end,:]
            spectrum.lineshape = spectrum.lineshape[start:end]  
        except:
            print(spectrum.number)
        
        if spectrum.lineshape.shape[0] < 1121:
            try:
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
            except:
                no_of_missing_values = 1121 - spectrum.lineshape.shape[0]
                
                no_values_begin = int((no_of_missing_values)/2)
                no_values_end = int((no_of_missing_values)/2)
                if no_of_missing_values %2 == 1:
                    no_values_end += 1
                
                begin_value = spectrum.lineshape[0]
                end_value = spectrum.lineshape[-1]

                spectrum.lineshape = np.concatenate(
                    (np.full(no_values_begin, begin_value),
                     spectrum.lineshape))
                
                spectrum.lineshape = np.concatenate(
                    (spectrum.lineshape,
                     np.full(no_values_end, end_value)))    
                        
        spectrum.update_range()
        spectrum.normalize()
        spectra.append(spectrum)
        index = filenames.index(name)
        X[index] = np.reshape(spectrum.lineshape, (-1,1))
        
        if plot_all:
            plt.plot(spectrum.x,spectrum.lineshape)
            plt.xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
            text = 'Spectrum no. ' + str(spectrum.number) + '\n' +\
                str(spectrum.label)
            plt.legend([text])
            plt.tight_layout()
            plt.show()
        
    return X, y, names

#%%
X,y, names = convert_spectra(plot_all = True)


output_file= r'C:\Users\pielsticker\Simulations\measured.h5py'
  
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('X', data = X,
                      compression="gzip", chunks=True,
                      maxshape=(None,X.shape[1],X.shape[2]))        
    hf.create_dataset('y', data = y,
                      compression="gzip", chunks=True,
                      maxshape=(None, y.shape[1]))
    hf.create_dataset('names', data=np.array(names, dtype='S'),
                      compression="gzip", chunks=True,
                      maxshape=(None, names.shape[1]))

    
#%% Check   
with h5py.File(output_file, 'r') as hf:
    dataset_size = hf['X'].shape[0]
    X_load = hf['X'][:,:,:]
    y_load = hf['y'][:,:]
    names_load = np.reshape(np.zeros(dataset_size),(-1,1))
    names_load_list = [name[0].decode("utf-8") for name in hf['names'][:,:]]
    names_load = np.reshape(np.array(names_load_list),(-1,1))
    print(np.allclose(X,X_load))
    print(np.allclose(y,y_load))
    #print(np.allclose(names,names_load))
