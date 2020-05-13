# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:25:02 2020

@author: pielsticker
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import fftconvolve
from base_model import Spectrum, MeasuredSpectrum, SimulatedSpectrum, \
    SyntheticSpectrum, Gauss, Lorentz


class Simulation():
    """
    """
    def __init__(self, input_spectra):   
        self.input_spectra = input_spectrum
        self.input_spectrum.nromalize()

        
        # The step width must be defined by the measured spectrum. 
        # All synthetic pectra need to have their step widths redefined
        # and their lineshapes rebuilt.
        self.start = self.input_spectrum.start 
        self.stop = self.input_spectrum.stop    
        self.step = self.input_spectrum.step 
        self.label = self.input_spectrum.label
   
        self.output_spectrum = SimulatedSpectrum(self.start,
                                                 self.stop,
                                                 self.step,
                                                 self.label)        
        self.output_spectrum.lineshape = self.input_spectrum.lineshape

    def shift_x(self, shift_x):
        """
        Shifts the ouput lineshape by some eV.
        Parameters
        ----------
        shift_x : int
            shift_x is in eV.
            shift_x has to be between -8 and 8 to be valid

        Returns
        -------
        None.

        """
        acceptable_values = list(range(-9, 9))
        
        if shift_x in acceptable_values:
            # scale the shift by the step size
            shift = int(np.round(shift_x/self.step, 1))
            
            len_y = len(self.output_spectrum.lineshape)
            begin_value = self.output_spectrum.lineshape[0]
            end_value = self.output_spectrum.lineshape[-1]

            if shift_x >= 0:
                self.output_spectrum.lineshape = np.concatenate(
                    (np.full(shift, begin_value),
                     self.output_spectrum.lineshape[:-shift]))
            else:
                self.output_spectrum.lineshape = np.concatenate(
                    (self.output_spectrum.lineshape[-shift:],
                     np.full(-shift, end_value)))
        
            self.output_spectrum.shift_x = shift_x
            
            # For normalization, take the sum of the original lineshape.
            b = np.nansum(self.input_spectrum.lineshape) 
            self.output_spectrum.lineshape /= b
        
        else:
            #return error and repeat input
            print('Shift value too big.')
            

    def scale_y(self, scale_y):    
        """
        Scales the intensity by a factor.
        Parameters
        ----------
        shift_x : int
            shift_x is in eV.
            shift_x has to be between -8 and 8 to be valid.

        Returns
        -------
        None.
        """
        self.output_spectrum.lineshape -= np.min(self.output_spectrum.lineshape)
        self.output_spectrum.lineshape = self.output_spectrum.lineshape*scale_y
        self.output_spectrum.scale_y = scale_y
        #b = np.nansum(self.input_spectrum.lineshape) 
        #self.output_spectrum.normalize()
    
    def add_noise(self, signal_to_noise = 1):
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
        
        intensity_max = np.max(self.output_spectrum.lineshape)
        noise = intensity_max/signal_to_noise
        
        poisson_noise = noise* np.random.poisson(1,
                                                 self.output_spectrum.lineshape.shape)
        print(set(poisson_noise))
        
        self.output_spectrum.lineshape = self.output_spectrum.lineshape + poisson_noise   
                                    
        self.output_spectrum.normalize()
        self.output_spectrum.signal_to_noise = signal_to_noise 
        
        
    def change_resolution(self, resolution):
        
        """
        Apply Gaussian instrumental broadening. This function broadens a spectrum
        assuming a Gaussian kernel. The width of the kernel is determined by the
        resolution. In particular, the function will determine the mean wavelength
        and set the Full Width at Half Maximum (FWHM) of the Gaussian to
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
        x = self.output_spectrum.x
        y = self.output_spectrum.lineshape
        
        fwhm = np.mean(x)/ float(resolution) 
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        # To preserve the position of spectral lines, the broadening function
        # must be centered at N//2 - (1-N%2) = N//2 + N%2 - 1
        lenx = len(x)
        step = self.output_spectrum.step
        gauss_x = (np.arange(lenx,
                             dtype=np.int) - sum(divmod(lenx, 2)) + 1) * step
        
        # The broadening spectrum is a synthetic spectrum.
        self.broadening_spectrum = SyntheticSpectrum(gauss_x[0],
                                                     gauss_x[-1],
                                                     step,
                                                     label = 'Gauss')
        self.add_broadening_peak(syn_spectrum = self.broadening_spectrum,
                                 peak_kind = 'Gauss',
                                 position = 0, 
                                 width = sigma)   
        self.broadening_spectrum.normalize()
        
        # This assures that the edges are handled correctly.
        len_y = len(y)
        y = np.concatenate((np.ones(len_y) * y[0],
                            y,
                            np.ones(len_y) * y[-1]))
        
        # This performs the convolution of the initial lineshape with
        # the Gaussian kernel.
        result = fftconvolve(y,
                             self.broadening_spectrum.lineshape,
                             mode="same")
        result = result[len_y:-len_y]
        
        self.output_spectrum.lineshape = result
        self.output_spectrum.normalize()
        self.output_spectrum.resolution = fwhm    

        
    def add_broadening_peak(self, syn_spectrum, peak_kind, position, width):
        if peak_kind == 'Gauss':
            syn_spectrum.addComponent(Gauss(position,width,1))
        elif peak_kind == 'Lorentz':
            syn_spectrum.addComponent(Lorentz(1,width,1))
                
      
        
    def plot_simulation(self):
        fig1, ax1 = plt.subplots(figsize=(5,4), dpi=100)
        fig1.patch.set_facecolor('0.9411')

        ax1.set_xlabel('Binding energy (eV)')
        ax1.set_ylabel('Intensity (arb. units)')
        ax1.set_title('Simulation')
        legend_list = ['original','simulated spectrum']
        
        
        for i in [self.input_spectrum,self.output_spectrum]:
            ax1.plot(np.flip(i.x), i.lineshape)
            
        ax1.legend(legend_list)
        fig1.tight_layout()
        
        try:
            broad_spec = self.broadening_spectrum
            fig2, ax2 = plt.subplots(figsize=(5,4), dpi=100)
            fig2.patch.set_facecolor('0.9411')
            
            ax2.set_xlabel('Binding energy (eV)')
            ax2.set_ylabel('Intensity (arb. units)')
            ax2.set_title('Broadening spectrum')
        
            ax2.plot(broad_spec.x, broad_spec.lineshape) 
            
            fig2.tight_layout()

        except:
            pass
    


#%% 
if __name__ == '__main__':
    label = 'Fe_metal'
    datapath = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'augmentation')[0] + '\\data'
                    
    filename = datapath + '\\' + label + '.txt'
        
    input_spec = MeasuredSpectrum(filename)
    
    sim = SpectrumChange(input_spec)
    #sim.shift_x(shift)
    #sim.scale_y(1)
    #sim.shift_x(-2)
    #sim.add_noise(signal_to_noise = 15)
    #sim.change_resolution(peak_kind = 'Gauss', width = 2)
    #sim.change_resolution(resolution = 250)
    sim.change_resolution(resolution = 250)
    sim.add_noise(signal_to_noise = 30)
    #sim.shift_x(2)
    #sim.shift_x(-2)

    sim.plot_simulation()    
        







