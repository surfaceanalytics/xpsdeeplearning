# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:47:32 2020

@author: pielsticker
"""

import os
import numpy as np
import json

try:
    from .spectra import SyntheticSpectrum
    from .peaks import Gauss, Lorentz, VacuumExcitation, Tougaard
except ImportError as e:
    if str(e) == "attempted relative import with no known parent package":
        pass
    else:
        raise
        #%%


class Scatterer:
    """
    Basic class for creating a scatterer which can be used as part of
    a ScatteringMedium.
    """

    def __init__(self, label):
        """
        Create a default loss function going from 0 to 200 eV with a 
        step size of 0.1 eV.

        Parameters
        ----------
        label : str
            String value for labeling the scatterer.
            Allowed values: 'default', 'He', 'H2', 'O2', 'N2', 

        Returns
        -------
        None.

        """
        self.label = label
        self.loss_function = SyntheticSpectrum(0, 200, 0.1, label="loss_fn")

        self.gas_diameter = 0.2  # In nanometers
        self.gas_cross_section = np.pi * (self.gas_diameter / 2) ** 2
        self.inelastic_xsect = 0.01  # in units of nm^3
        self.norm_factor = 1

    def build_loss_from_json(self, input_datapath):
        """
        This function builds the spectrum of the loss function from the
        components in a scatterrer loaded from a json file.             

        Parameters
        ----------
        input_datapath : str
            Filepath of the json file.

        Returns
        -------
        None.

        """
        with open(input_datapath, "r") as json_file:
            test = json.load(json_file)

        scatterer_dict = test[self.label]
        self.inelastic_xsect = scatterer_dict["inelastic_xsect"]
        self.norm_factor = scatterer_dict["norm_factor"]

        for i in scatterer_dict["loss_function"]:
            if i["type"] == "Gauss":
                self.loss_function.add_component(
                    Gauss(
                        i["params"]["position"],
                        i["params"]["width"],
                        i["params"]["intensity"],
                    ),
                    rebuild=False,
                )
            elif i["type"] == "Lorentz":
                self.loss_function.add_component(
                    Lorentz(
                        i["params"]["position"],
                        i["params"]["width"],
                        i["params"]["intensity"],
                    ),
                    rebuild=False,
                )
            elif i["type"] == "VacuumExcitation":
                self.loss_function.add_component(
                    VacuumExcitation(
                        i["params"]["edge"],
                        i["params"]["fermi_width"],
                        i["params"]["intensity"],
                        i["params"]["exponent"],
                    ),
                    rebuild=False,
                )
            elif i["type"] == "Tougaard":
                self.loss_function.add_component(
                    Tougaard(
                        i["params"]["B"],
                        i["params"]["C"],
                        i["params"]["D"],
                        i["params"]["Eg"],
                    ),
                    rebuild=False,
                )
        self.loss_function.rebuild()


class ScatteringMedium:
    """
    A ScatteringMedium contains one or more Scatterer object and can be
    used to simulate scattering of photoelectrons in a medium.
    """

    def __init__(self, label):
        """
        In this case, only one Scatterer object is used, 
        depending on the input label.

        Parameters
        ----------
        label : str
            String value for choosing the scatterer.
            Allowed values: 'default', 'He', 'H2', 'O2', 'N2'

        Returns
        -------
        None.

        """
        self.scatterer = Scatterer(label)
        self.R = 8.314463e25  # Gas constant in nm^3.mbar.K^-1.mol^-1
        self.avagadro = 6.022141e23  # Avagadro contant
        self.T = 300  # Temperature in Kelvin
        self.pressure = 1  # In mbar
        self.distance = 0.80  # In millimeters
        self.calc_density()

    def calc_density(self):
        """
        Calculate the molecular density in units of particles per nm^3

        Returns
        -------
        None.

        """
        self.density = self.pressure / (self.R * self.T) * self.avagadro

    def convert_distance(self):
        """
        Calculate the distance from mm to nm.

        Returns
        -------
        None.

        """
        self.distance *= 1000000


#%%
if __name__ == "__main__":
    from spectra import SyntheticSpectrum
    from peaks import Gauss, Lorentz, VacuumExcitation, Tougaard
    from figures import Figure

    label = "He"
    distance = 1
    pressure = 2
    medium = ScatteringMedium(label)
    medium.scatterer.step = 0.1
    medium.calcDensity()

    input_datapath = (
        os.path.dirname(os.path.abspath(__file__)).partition("augmentation")[
            0
        ]
        + "\\data\\scatterers.json"
    )
    medium.scatterer.build_loss_from_json(input_datapath)
    loss_fn = medium.scatterer.loss_function
    loss_x = loss_fn.x
    loss_lineshape = loss_fn.lineshape

    loss_fn = medium.scatterer.loss_function
    loss_lineshape = loss_fn.lineshape

    # Plot the loss function up to 200 eV.
    figure = Figure(loss_fn.x, loss_lineshape, title="loss function")
    figure.ax.set_xlim(left=np.min(loss_fn.x), right=200)
