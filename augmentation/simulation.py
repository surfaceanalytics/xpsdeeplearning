# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:25:02 2020

@author: pielsticker
"""

import numpy as np
import os
from base_model import MeasuredSpectrum, SimulatedSpectrum, Figure


class Simulation():
    """
    Basic class for simulating a spectrum from input spectra.
    The main methods are linear combination of the input spectra and
    changes to the resolution, S/N ratio and x-axis of a spectrum.
    """
    def __init__(self, input_spectra):  
        """
        Initialize the input spectra (a list) and and empty 
        SimulatedSpectrum for the output. The x-range for the output
        spectrum is originally the same as the first input spectrum.

        Parameters
        ----------
        input_spectra : list
            List of instances of the MeasuredSpectrum class.

        Returns
        -------
        None.

        """
        self.input_spectra = input_spectra
        for spectrum in input_spectra:
            spectrum.normalize()
        
        # Initilaize the axes and label to the spectrum loaded first.
        start = self.input_spectra[0].start 
        stop = self.input_spectra[0].stop    
        step = self.input_spectra[0].step 
        label = self.input_spectra[0].label
        
        self.output_spectrum = SimulatedSpectrum(start,
                                                 stop,
                                                 step,
                                                 label)         
       
        
    def combine_linear(self, scaling_params):
        """
        Performs a linear combination of the input spectra. Each
        spectrum is scaled by a parameter in the range of [0,1]. All
        scaling parameter have to add up to 1.

        Parameters
        ----------
        scaling_params : list
            Parameter list of float values to scale the input spectra.
            The length of the scaling_params list has to be the same
            as the number of spectra for the linear combination.

        Returns
        -------
        None.

        """
        # Make sure that the right amount of params is given.
        if len(self.input_spectra) < len(scaling_params):
            print('Please supply the correct amount of scaling parameters.')
            print('Simulated spectrum was not changed!') 
            
        elif len(self.input_spectra) > len(scaling_params):
            print('Please supply enough scaling parameters.')
            print('Simulated spectrum was not changed!') 
         
        else:
            self.output_spectrum.label = {}
            if np.round(sum(scaling_params),decimals = 1) == 1.0:
                output_list = []
                for i in range(len(self.input_spectra)):
                    # Species = List of input spectra names
                    species = list(self.input_spectra[i].label.keys())[0]
                    concentration = scaling_params[i]
                    
                    intensity = self.input_spectra[i].lineshape* \
                                scaling_params[i]
                    output_list.append(intensity)
                    
                    # For each species, the label gets a new key:value
                    # pair of the format species: concentration
                    if concentration == 0:
                        pass
                    else:
                        self.output_spectrum.label[species] = concentration
           
                # Linear combination
                self.output_spectrum.lineshape = sum(output_list)
                self.output_spectrum.normalize()
                
            else:
               print('Scaling parameters have to sum to 1!') 
               print('Simulated spectrum was not changed!') 


    def change_spectrum(self, spectrum = None, **kwargs, ):
        """
        Parameters
        ----------
        spectrum : Spectrum, optional
            A Spectrum object can be supplied if one wants to change a
            single inout spectrum and not change a spectrum that was 
            already created using a linear combination. 
            If spectrum == None,then the current output spectrum is
            changed. The default is None.
        **kwargs : str
            resolution: int
                To perform a convolution of the spectrum with a
                gaussian with FWHM = resolution/mean(x) where x is the 
                x-axis of the spectrum.
            signal_to_noise: int
                To add poisson-distributed noise at to the spectrum. 
                Signal-to-noise describes the S/N ratio of the
                resulting spectrum.
            shift_x: int
                To shift the spectrum by some eV.
            
        Returns
        -------
        None.

        """
        if spectrum != None:
            # The step width is defined by the measured spectrum. 
            # The output spectrum needs to have its step widths 
            # redefined.
            self.output_spectrum.lineshape = spectrum.lineshape
            start = self.spectrum.start 
            stop = self.spectrum.stop    
            step = self.spectrum.step 
            self.label = self.spectrum.label
            self.output_spectrum.x = np.flip(np.arange(
                                                 start,
                                                 stop+step,
                                                 step))
        else:
            pass
            
        if 'fwhm' in kwargs.keys():
            self.output_spectrum.resolution = kwargs['fwhm']
            self.output_spectrum.change_resolution(kwargs['fwhm'])
            
        if 'shift_x' in kwargs.keys():
            self.output_spectrum.shift_x = kwargs['shift_x']
            self.output_spectrum.shift_horizontal(kwargs['shift_x'])
            
        if 'signal_to_noise' in kwargs.keys():
            self.output_spectrum.signal_to_noise = kwargs['signal_to_noise'] 
            self.output_spectrum.add_noise(kwargs['signal_to_noise'])
            
        self.output_spectrum.normalize()


    def plot_simulation(self, plot_inputs = False):   
        """
        Creates Figure objects for the output pectrum and (optionally)
        for the input spectra.

        Parameters
        ----------
        plot_inputs : bool, optional
            If plot_inputs == True, the input spectra are also plotted.
            Otherwise, only the output spectrum is plotted.
            The default is False.

        Returns
        -------
        None.

        """        
        if plot_inputs:
            figs_input = []
            for spectrum in self.input_spectra:
                x = spectrum.x
                y = spectrum.lineshape
                title = next(iter(spectrum.label))
                fig_input = Figure(x, y, title = title)
                figs_input.append(fig_input)
        
        Figure(self.output_spectrum.x,
               self.output_spectrum.lineshape,
               title = self.output_spectrum.type)

            

#%% 
if __name__ == '__main__':
    datapath = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'augmentation')[0] + '\\data' + '\\measured'
       
    labels = ['Fe_metal','FeO','Fe3O4','Fe2O3']
    
    input_spectra = []
    for label in labels:
        filename = datapath + '\\' + label + '.txt'
        input_spectra += [MeasuredSpectrum(filename)]
        
    del(datapath,label,labels,filename)
  
    sim = Simulation(input_spectra)
    sim.combine_linear(scaling_params = [0.4,0.4,0.1,0.1])                

    sim.change_spectrum(shift_x = 2,
                        signal_to_noise = 150,
                        fwhm = 1050)
    
    print('Linear combination parameters: ' + str(sim.output_spectrum.label))
    sim.plot_simulation(plot_inputs = True)





