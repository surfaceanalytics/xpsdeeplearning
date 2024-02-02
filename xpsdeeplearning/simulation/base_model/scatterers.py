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
Basic definition of scattering objects for simulating spectra.
"""
import json
import numpy as np

from xpsdeeplearning.simulation.base_model.converters.peaks import (
    Gauss,
    Lorentz,
    VacuumExcitation,
    Tougaard,
)
from xpsdeeplearning.simulation.base_model.spectra import SyntheticSpectrum


class Scatterer:
    """Basic class for creating a scatterer."""

    def __init__(self, label):
        """
        Create a default loss function.

        The default goes from 0 to 200 eV with a
        step size of 0.1 eV.

        Parameters
        ----------
        label : str
            String value for labeling the scatterer.
            Allowed values: 'default', 'He', 'H2', 'O2', 'N2'.

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
        Build the spectrum of the loss function.

        The components are taken from a scatterer loaded from a JSON
        file.

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
    """A scattering medium to simulate scattering of photoelectrons."""

    def __init__(self, label):
        """
        Initialize the medium with one scatterer.

        A ScatteringMedium contains one or more Scatterer objects.
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
        Calculate the molecular density in units of particles per nm^3.

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
