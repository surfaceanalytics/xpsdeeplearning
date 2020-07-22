# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:39:34 2020

@author: pielsticker
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve


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
            l = self.intensity * \
                1 / (1 + ((self.position-x)/(self.width/2))**2)
            return l
    
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
        self.normalize()
    
    def convert(self, filename):
        file = open(filename,'r')
        lines = []
        for line in file.readlines():
            lines += [line]
        # This takes the species given in the first line
        species = str(lines[0]).split('\n')[0] 
        lines = lines[1:]
        lines = [[float(i) for i in line.split()] for line in lines]
        data = np.array(lines)
        # The label is a dictionary of the form
        # {species: concentration}.
        label = {species: 1.0}
        
        return label, data


class SyntheticSpectrum(Spectrum):
    def __init__(self,start,stop,step,label):
        Spectrum.__init__(self,start,stop,step,label)
        self.type = 'synthetic'
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
        self.type = 'simulated'
        self.shift_x = None
        self.signal_to_noise = None
        self.resolution = None
    
    def shift_horizontal(self, shift_x):
        """
        Shifts the output lineshape by some eV.
        Parameters
        ----------
        shift_x : int
            shift_x is in eV.
            shift_x has to be between -8 and 8 to be valid

        Returns
        -------
        None.

        """
        b = np.nansum(self.lineshape) 
        
        acceptable_values = [-9, 9]
        
        if shift_x == None:
            pass
        elif (shift_x >= acceptable_values[0] \
              and shift_x <= acceptable_values[1]):
            # scale the shift by the step size
            shift = int(np.round(shift_x/self.step, 1))
            
            begin_value = self.lineshape[0]
            end_value = self.lineshape[-1]

            if shift_x < 0:
                self.lineshape = np.concatenate(
                    (np.full(-shift, begin_value),
                     self.lineshape[:shift]))
                
            elif shift_x > 0:
                self.lineshape = np.concatenate(
                    (self.lineshape[shift:],
                     np.full(shift, end_value)))      
            else:
                pass
        
            #self.output_spectrum.shift_x = shift_x
            
            # For normalization, take the sum of the original lineshape.
            if b !=0:
                self.lineshape /= b
            else:
                print("Simulation was not successful.")
        
        else:
            #return error and repeat input
            print('Shift value too big.')
            print("Simulated spectrum was not changed!") 
    


    def add_noise(self, signal_to_noise):
        """
        Adds noise from a Poisson distribution.
        
        Parameters
        ----------
        signal_to_noise : int
            Integer value for the amount of noise to be added.   
            
        Returns
        -------
        None.
        """
        if (signal_to_noise == 0 or signal_to_noise == None):
            pass
        else:
            intensity_max = np.max(self.lineshape)
            noise = intensity_max/signal_to_noise
        
            poisson_noise = noise* np.random.poisson(1,
                                                     self.lineshape.shape)
        
            self.lineshape = self.lineshape + poisson_noise                         
            self.normalize()

                    
        
    def change_resolution(self, resolution):
        """
        Apply Gaussian instrumental broadening. This function broadens
        a spectrum assuming a Gaussian kernel. The width of the kernel
        is determined by the resolution. In particular, the function
        will determine the mean wavelengt and set the Full Width at
        Half Maximum (FWHM) of the Gaussian to
        (mean wavelength)/resolution. 
    
        Parameters
        ----------
        resolution : int
            The spectral resolution.

        Returns
        -------
        Broadened spectrum : array
            The input spectrum convolved with a Gaussian
            kernel.
        FWHM : float, optional
            The Full Width at Half Maximum (FWHM) of the
            used Gaussian kernel.
        """
        x = self.x
        y = self.lineshape
        

        if (resolution == 0 or resolution == None):
            fwhm = resolution
        else:
            fwhm = np.mean(x)/ float(resolution) 
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
            # To preserve the position of spectral lines, the broadening
            # function must be centered at N//2 - (1-N%2) = N//2 + N%2 - 1.
            lenx = len(x)
            step = self.step
            gauss_x = (np.arange(
                lenx, dtype=np.int) - sum(divmod(lenx, 2)) + 1) * step
        
        
            # The broadening spectrum is a synthetic spectrum.
            broadening_spectrum = SyntheticSpectrum(gauss_x[0],
                                                    gauss_x[-1],
                                                    step,
                                                    label = 'Gauss')
            broadening_spectrum.addComponent(Gauss(position = 0,
                                                   width = sigma,
                                                   intensity = 1))
                     
            # This assures that the edges are handled correctly.
            len_y = len(y)
            y = np.concatenate((np.ones(len_y) * y[0],
                                y,
                                np.ones(len_y) * y[-1]))
        
            # This performs the convolution of the initial lineshape with
            # the Gaussian kernel.
            result = fftconvolve(y,
                                 broadening_spectrum.lineshape,
                                 mode="same")
            result = result[len_y:-len_y]
            
            self.lineshape = result
            self.normalize()
        self.fwhm = fwhm    
        
class Figure:

    def __init__(self, x, y, title):
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(5,4), dpi=100)
        self.fig.patch.set_facecolor('0.9411')
        self.ax.plot(x, y)
        self.ax.set_xlabel('Binding energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')
        self.ax.set_xlim(left=np.max(x), right=np.min(x))
        self.ax.set_title(title)
        self.fig.tight_layout()

#%% 
if __name__ == '__main__':
    label = 'Fe2O3'
    datapath = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'augmentation')[0] + 'data\\references'
                    
    filename = datapath + '\\' + label + '.txt'
        
    spec = MeasuredSpectrum(filename)
    
    fig = Figure(spec.x, spec.lineshape, title = label)
    begin_value = spec.lineshape[0]
