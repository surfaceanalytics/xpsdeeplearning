# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:39:34 2020

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
        self.x = np.flip(np.arange(self.start,self.stop+self.step,self.step))
        self.clear_lineshape()
        self.label = label
        
    def clear_lineshape(self):
        self.lineshape = np.zeros(int(np.round((self.stop-self.start+self.step)/self.step, decimals = 1)))
        self.x = np.flip(np.arange(self.start,self.stop+self.step,self.step))


    def normalize(self):
        if np.sum(self.lineshape) != 0:
            self.lineshape = self.lineshape / np.nansum(self.lineshape) 
        
class Peak:
    def __init__(self, position, width, intensity):
        self.position = position
        self.width = width
        self.intensity = intensity
        
class Gauss(Peak):
    def __init__(self,position,width,intensity):
        Peak.__init__(self, position, width, intensity)

    def function(self, x):
        if self.width != 0:
            g = self.intensity / (self.width * np.sqrt(2 * np.pi)) \
            * np.exp(-0.5 * ((x-self.position)/self.width)**2)
            return g

class Lorentz(Peak):
    def __init__(self,position,width,intensity):
        Peak.__init__(self, position, width, intensity)    
    
    def function(self, x):
        if self.width != 0:
            l = self.intensity * 1 / (1 + ((self.position-x)/(self.width/2))**2)
            return l
    
class MeasuredSpectrum(Spectrum):
    def __init__(self, filename):
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
    
    def convert(self, filename):
        file = open(filename,'r')
        lines = []
        for line in file.readlines():
            lines += [line]
        label = str(lines[0]) # this takes the species given in the first line
        lines = lines[2:]
        lines = [[float(i) for i in line.split()] for line in lines]
        data = np.array(lines)
        
        return label, data


class SyntheticSpectrum(Spectrum):
    def __init__(self,start,stop,step,label):
        Spectrum.__init__(self,start,stop,step,label)
        self.components = []
    
    def build_line(self):
        self.clear_lineshape()
        for component in self.components:
            y = np.array([component.function(x) for x in self.x])
            self.lineshape = np.add(self.lineshape,y)
            
    def addComponent(self,component):
        self.components += [component]
        self.rebuild()   
        
    def remove_component(self, comp_idx):
        del self.components[comp_idx]
        self.rebuild()

    def rebuild(self):
        self.update_range()
        self.build_line()
        
    def update_range(self):
        self.x = np.flip(np.arange(self.start,self.stop+self.step,self.step))


class SimulatedSpectrum(Spectrum):
    def __init__(self,start,stop,step,label):
        Spectrum.__init__(self,start,stop,step,label)
        self.shift_x = None
        self.scale_y = None
        self.signal_to_noise = None
        self.resolution = None


#%% 
if __name__ == '__main__':
    label = 'FeO'
    datapath = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'augmentation')[0] + '\\data'
                    
    filename = datapath + '\\' + label + '.txt'
        
    spec = MeasuredSpectrum(filename)
    
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)
    fig.patch.set_facecolor('0.9411')

    ax.set_xlabel('Binding energy (eV)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title(label)
    ax.plot(np.flip(spec.x), spec.lineshape)

            
