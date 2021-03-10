# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:01:19 2020

@author: pielsticker
"""
import numpy as np
import warnings

#%%
class Peak:
    """
    Basic class for a peak that gets a number of parameters and creates a 
    lineshape based on these parameters.
    """
    def __init__(self, 
                 position,
                 width, 
                 intensity):
        self.position = position
        self.width = width
        self.intensity = intensity
        

class Gauss(Peak):
    """
    Gaussian peak with position, width, and intensity parameters.
    """
    def __init__(self,
                 position,
                 width,
                 intensity):
        """
        Initialize basic Peak class. 

        Parameters
        ----------
        position : float
            Position of the main peak.
        width : float
            FWHM of the main peak.
        intensity : float
            Intensity of the main peak.
            
        Returns
        -------
        None.

        """
        super(Gauss, self).__init__(position, width, intensity)

    def function(self, 
                 x):
        """
        Create a numpy array of a Gaussian peak based on the
        input x values. 

        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        v : arr
            Peak lineshape.

        """
        if self.width != 0:
            g = self.intensity / (self.width * np.sqrt(2 * np.pi)) \
            * np.exp(-0.5 * ((x-self.position)/self.width)**2)
            return g


class Lorentz(Peak):
    """
    Lorentzian peak with position, width, and intensity parameters.
    """
    def __init__(self,
                 position,
                 width,
                 intensity):
        """
        Initialize basic Peak class. 

        Parameters
        ----------
        position : float
            Position of the main peak.
        width : float
            FWHM of the main peak.
        intensity : float
            Intensity of the main peak.

        Returns
        -------
        None.

        """
        super(Lorentz, self).__init__(position, width, intensity) 
    
    def function(self,
                 x):
        """
        Create a numpy array of a Lorentzian peak based on the
        input x values. 

        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        v : arr
            Peak lineshape.

        """
        if self.width != 0:
            l = self.intensity * \
                1 / (1 + ((self.position-x)/(self.width/2))**2)
            return l


class Voigt(Peak):
    """
    Voigt peak shape with position, width, and intensity parameters.
    """    
    def __init__(self,
                 position,
                 width, 
                 intensity, 
                 fraction_gauss = 0.5):
        """
        Initialize basic Peak class. 

        Parameters
        ----------
        position : float
            Position of the main peak.
        width : float
            FWHM of the main peak.
        intensity : float
            Intensity of the main peak.
        fraction_gauss : float, optional
            Ratio between Gaussian and Lorentzian contribution.
            The default is 0.5.

        Returns
        -------
        None.

        """
        super(Voigt, self).__init__(position, width, intensity) 
        self.fraction_gauss = fraction_gauss
        
    def function(self,
                 x):
        """
        Create a numpy array of a mixed Gaussian/Lorentzian peak
        based on the input x values. 

        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        v : arr
            Peak lineshape.

        """
        if self.width != 0:
            v = ((self.fraction_gauss 
                 * Gauss(self.position,
                         self.width,
                         self.intensity).function(x))
                + ((1- self.fraction_gauss)
                * Lorentz(self.position,
                          self.width,
                          self.intensity).function(x)))
            return v
  
        
class VacuumExcitation():
    """
    Class for simulating a broad vacuum excitation starting at the 
    fermi edge and using a power law to simulate the extended 
    background shape.
    """
    def __init__(self,
                 edge, 
                 fermi_width, 
                 intensity,
                 exponent):
        """
        
        Parameters
        ----------
        edge : float
            Position of the Fermi edge.
        fermi_width : float
            Width of the Fermid edge.
        intensity : float
            Intensity of the vacuum excitation.
        exponent : float
            Exponent of the power law.

        Returns
        -------
        None.

        """
        self.edge = edge
        self.fermi_width = fermi_width
        self.intensity = intensity
        self.exponent = exponent
        
    def fermi_edge(self, 
                   x):
        """
        Create the FE lineshape.
        
        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        f : arr
            Fermi edge lineshape.

        """
        k = 0.1
        f = 1/(np.exp((self.edge-x)/(k*self.fermi_width))+1)
        return f
    
    def power_law(self, 
                  x):
        """
        Create the lineshape away from the FE following a power law.
        
        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        f : arr
            Power law lineshape.

        """
        p = np.exp(-1 * (x+self.edge) * self.exponent)
        return p
    
    def function(self,
                 x):
        """
        Combine the fermi edge and power law lineshapes and scale
        by the overall intensity.
        
        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        f : arr
            Output lineshape of the vacuum excitation.

        """
        if self.fermi_width !=0:
            f = (self.fermi_edge(x)) * self.power_law(x) * self.intensity
            return f 
    
    
class Tougaard():
    """
    Class for simulating a Tougaard background lineshape.
    """
    def __init__(self,
                 B,
                 C,
                 D,
                 Eg):
        """
        U4 Tougaard lineshape based on 4 parameters.

        Parameters
        ----------
        B : float
        C : float
        D : float
        Eg : float

        Returns
        -------
        None.

        """
        
        self.B = B
        self.C = C
        self.D = D
        self.Eg = Eg
        self.t = 300 # Temperature in Kelvin
        self.kb = 0.000086 # Boltzman constant
        
    def function(self, 
                 x):
        """
        Create the Tougaard lineshape of the form
        F(x) = B*x/(C-x**2)**2 + D*x**2) if x > Eg
        F(x) = 0 <= Eg
        
        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        f : arr
            Fermi edge lineshape.

        """
        warnings.simplefilter("ignore")
        C = self.C * 20
        f = ((self.B * x) / ((C-x**2)**2 + self.D*x**2)
        * 1/(np.exp((self.Eg - x)/(self.t * self.kb)) + 1))
        
        return f