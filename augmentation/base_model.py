# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:39:34 2020

@author: pielsticker
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
import warnings

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

class Voigt(Peak):
    def __init__(self, position, width, intensity, fraction_gauss=0.5):
        Peak.__init__(self, position, width, intensity)
        self.fraction_gauss = fraction_gauss
        
    def function(self,x):
        if self.width != 0:
            v = ((self.fraction_gauss 
                 * Gauss(self.position,self.width,self.intensity).function(x))
                + ((1- self.fraction_gauss)
                * Lorentz(self.position,self.width,self.intensity).function(x)))
            return v
        
class VacuumExcitation():
    def __init__(self, edge, fermi_width, intensity, exponent):
        self.edge = edge
        self.fermi_width = fermi_width
        self.intensity = intensity
        self.exponent = exponent
        
    def Fermi(self, x):
        k = 0.1
        f = 1/(np.exp((self.edge-x)/(k*self.fermi_width))+1)
        return f
    
    def Power(self, x):
        p = np.exp(-1 * (x+self.edge) * self.exponent)
        return p
    
    def function(self,x):
        if self.fermi_width !=0:
            f = (self.Fermi(x)) * self.Power(x) * self.intensity
            return f 
    
class Tougaard():
    def __init__(self,B, C, D, Eg):
        self.B = B
        self.C = C
        self.D = D
        self.Eg = Eg
        #self.t = 300 # Temperature in Kelvin
        #self.kb = 0.000086 # Boltzman constant
        
    def function(self, x):
        warnings.simplefilter("ignore")
        kb = 0.000086
        t = 300
        C = self.C * 20
        f = ((self.B * x) / ((C-x**2)**2 + self.D*x**2)
        * 1/(np.exp((self.Eg - x)/(t * kb)) + 1))
        
        return f

    
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
            
    def addComponent(self,component, rebuild=True):
        self.components += [component]
        if rebuild == True:
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
        
        
    def scatter_in_gas(self, filetype = 'json',
                       label = 'He', distance = 0.8, pressure = 1.0):
        """
        This method is for the case of scattering though a gas phase/film.
        First the convolved spectra are dotted with the factors vector to
        scale all the inelastic scatterd spectra by the Poisson and angular
        factors. Then the elastically scattered and non-scattered spectra
        are scaled by their respective factors.

        Parameters
        ----------
        label : str
            DESCRIPTION. The default is 'He'.
        distance : float
            Distance in mm. The default is '0.8'.
        pressure : float
            Pressure in mbar. The default is 1.

        Returns
        -------
        None.

        """
        if label in ['He', 'H2', 'N2', 'O2']:
            medium = ScatteringMedium(label)
            medium.scatterer.loss_function.step = self.step
            medium.pressure = pressure
            medium.distance = distance
            medium.convert_distance()
            medium.calcDensity()   
            
            if filetype == 'json':
                medium.scatterer.build_loss_from_json()
                loss_fn = medium.scatterer.loss_function
                loss_x = loss_fn.x
                loss_lineshape = loss_fn.lineshape                     
                        
            elif filetype == 'csv':
                root_dir = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\data'
                #os.path.abspath(__file__)).partition(
                #    'augmentation')[0] + '\\data'
                data_dir = os.path.join(root_dir,
                                          label + ' loss function.csv')
                loss_x = []
                loss_lineshape = []
                
                with open(data_dir, mode='r') as csv_file:
                    reader = csv.DictReader(csv_file, fieldnames = ['x','y'])
                    for row in reader:
                        d = dict(row)
                        loss_x.append(float(d['x']))
                        loss_lineshape.append(float(d['y']))
                        
                loss_x = np.array(loss_x)
                loss_lineshape = np.array(loss_lineshape)
                
                medium.scatterer.inelastic_xsect = 0.08
                medium.scatterer.norm_factor = 1
            
            if np.sum(loss_lineshape) != 0:
                loss_lineshape /= np.sum(loss_lineshape)  
                               
# =============================================================================
#             figure = Figure(loss_x, loss_lineshape, 
#                             title = 'Loss function for ' + label)
#             figure.ax.set_xlim(left=np.min(loss_x), right=100)                    
# =============================================================================
                
            y = self.lineshape
    
            min_value = np.min(y)
            y -= min_value
           
            #loss = np.flip(loss_lineshape)
            loss = loss_lineshape

            ''' Pad the imput spectrum with zeros so that it has the same
            dimensions as the loss function.
            '''
            input_spec_padded = np.pad(y, (loss.shape[0]-y.shape[0],0),
                                       'constant', constant_values = 0)
            
            ''' Take Fourier transform of the input spectrum.
            '''
            fft_input_spec = np.fft.fft(input_spec_padded)
            ''' 
            Take the Fourier transform of the loss function.
            '''
            fft_loss = np.fft.fft(loss)
            poisson_factor = medium.distance * medium.scatterer.inelastic_xsect * medium.density
            norm_factor = medium.scatterer.norm_factor
            total_factor = poisson_factor * norm_factor       
            
            exp_factor = np.exp(-1 * poisson_factor)
            
            fft_total = exp_factor * np.multiply(fft_input_spec, 
                                         np.exp(total_factor*fft_loss))
                   
            ''' Take the inverse Fourier transform of the convolved spectrum.
            '''
            total = np.real(np.fft.ifft(fft_total)[-len(y):])
            result = total + min_value  

            self.lineshape = result
            self.normalize()
            self.scatterer = label
            self.pressure = pressure
            self.distance = distance

        elif label == None:
            pass
        else:
            print('Please enter a valid scatterer label!')
 
        
class Scatterer():
    def __init__(self, label):
        self.label = label    
        self.loss_function = SyntheticSpectrum(0,600,0.1, label = 'loss_fn')
        
        self.gas_diameter = 0.2 #In nanometers
        self.gas_cross_section = np.pi * (self.gas_diameter / 2)**2
        self.inelastic_xsect = 0.01 # in units of nm^3
        self.norm_factor = 1 
        
    def build_loss_from_json(self):
        """ This function builds the spectrum of the loss function from the
        components in a scatterrer loaded from JSON
        """     
        input_datapath = os.path.dirname(
        os.path.abspath(__file__)).partition(
            'augmentation')[0] + '\\data\\scatterers.json'
                
        #input_datapath = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\data\scatterers.json'
        
        with open(input_datapath, 'r') as json_file:
            test = json.load(json_file)
        
        scatterer_dict = test[self.label]
        self.inelastic_xsect = scatterer_dict['inelastic_xsect']
        self.norm_factor = scatterer_dict['norm_factor']  

        for i in scatterer_dict['loss_function']:
            if i['type'] == 'Gauss':
                self.loss_function.addComponent(
                        Gauss(i['params']['position'], i['params']['width'], 
                        i['params']['intensity']), rebuild = False)
            elif i['type'] == 'Lorentz':
                self.loss_function.addComponent(
                        Lorentz(i['params']['position'], i['params']['width'], 
                        i['params']['intensity']), rebuild = False)
            elif i['type'] == 'VacuumExcitation':
                self.loss_function.addComponent(
                        VacuumExcitation(
                        i['params']['edge'], i['params']['fermi_width'], 
                        i['params']['intensity'], i['params']['exponent']), 
                        rebuild = False)
            elif i['type'] == 'Tougaard':
                self.loss_function.addComponent(
                        Tougaard(
                        i['params']['B'], i['params']['C'], 
                        i['params']['D'], i['params']['Eg']), rebuild = False)
        self.loss_function.rebuild()                          
        
class ScatteringMedium():
    def __init__(self, label):
        self.scatterer = Scatterer(label)
        self.R = 8.314463E+25 # gas constant in units of nm^3.mbar.K^-1.mol^-1
        self.avagadro = 6.022141E+23 # Avagadro's contant
        self.T = 300 # temperature in Kelvin
        self.pressure = 1 # In mbar
        self.distance = 0.80 # In millimeters
        self.calcDensity()

        
    def calcDensity(self):
         # molecular density in units of particles per nm^3
        self.density = self.pressure / (self.R * self.T) * self.avagadro 
        
    def convert_distance(self):
        self.distance *= 1000000
        
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


    label = 'He'
    distance  = 1
    pressure = 2
    medium = ScatteringMedium(label)
    medium.scatterer.step = 0.1
    medium.calcDensity()
    
    medium.scatterer.build_loss_from_json()
    loss_fn = medium.scatterer.loss_function
    loss_x = loss_fn.x
    loss_lineshape = loss_fn.lineshape
    
    loss_fn = medium.scatterer.loss_function
    loss_lineshape = loss_fn.lineshape
    
    figure = Figure(loss_fn.x, loss_lineshape, title = 'loss function')
    figure.ax.set_xlim(left=np.min(loss_fn.x),right=200)
