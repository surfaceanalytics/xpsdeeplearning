#
# Copyright the xpsdeeplearning authors.
#
# This file is part of xpsdeeplearning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Basic definition of peak shapes for simulating spectra.
"""
import warnings
import numpy as np


class Peak:
    """Basic class for a peak."""

    def __init__(self, position, width, intensity):
        self.position = position
        self.width = width
        self.intensity = intensity


class Gauss(Peak):
    """Gaussian peak with position, width, and intensity."""

    def function(self, x):
        """
        Create a numpy array of a Gaussian peak.

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
            gaussian = (
                self.intensity
                / (self.width * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - self.position) / self.width) ** 2)
            )
            return gaussian
        return None


class Lorentz(Peak):
    """Lorentzian peak with position, width, and intensity."""

    def function(self, x):
        """
        Create a numpy array of a Lorentzian peak.

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
            lorentzian = (
                self.intensity * 1 / (1 + ((self.position - x) / (self.width / 2)) ** 2)
            )
            return lorentzian
        return None


class Voigt(Peak):
    """Voigt peak shape with position, width, and intensity."""

    def __init__(self, position, width, intensity, fraction_gauss=0.5):
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
        super().__init__(position, width, intensity)
        self.fraction_gauss = fraction_gauss

    def function(self, x):
        """
        Create a numpy array of a mixed Gaussian/Lorentzian peak.

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
            voigt = (
                self.fraction_gauss
                * Gauss(self.position, self.width, self.intensity).function(x)
            ) + (
                (1 - self.fraction_gauss)
                * Lorentz(self.position, self.width, self.intensity).function(x)
            )
            return voigt
        return None


class VacuumExcitation:
    """Class for a broad vacuum excitation."""

    def __init__(self, edge, fermi_width, intensity, exponent):
        """
        Initiliaze the four defining parameters.

        A vacuum excitation is simulated starting at the fermi edge
        and using a power law to simulate the extended background
        shape.

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

    def fermi_edge(self, x):
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
        fermi = 1 / (np.exp((self.edge - x) / (k * self.fermi_width)) + 1)
        return fermi

    def power_law(self, x):
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
        power_law = np.exp(-1 * (x + self.edge) * self.exponent)
        return power_law

    def function(self, x):
        """
        Combine the fermi edge and power law lineshapes.

        The lineshape is scaled by the overall intensity.

        Parameters
        ----------
        x : arr
            Numpy array containing the x axis of the peak.

        Returns
        -------
        f : arr
            Output lineshape of the vacuum excitation.

        """
        if self.fermi_width != 0:
            vac_exc = (self.fermi_edge(x)) * self.power_law(x) * self.intensity
            return vac_exc
        return None


class Tougaard:
    """Class for simulating a Tougaard background lineshape."""

    def __init__(self, B, C, D, Eg):
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
        self.t = 300  # Temperature in Kelvin
        self.kb = 0.000086  # Boltzman constant

    def function(self, x):
        """
        Create the Tougaard lineshape.

        The lineshape is of the form
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
        tougaard = (
            (self.B * x)
            / ((C - x**2) ** 2 + self.D * x**2)
            * 1
            / (np.exp((self.Eg - x) / (self.t * self.kb)) + 1)
        )

        return tougaard
