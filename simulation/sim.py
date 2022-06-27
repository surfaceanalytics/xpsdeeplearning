# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:25:02 2020.

@author: pielsticker
"""

import numpy as np
import os
from base_model.spectra import MeasuredSpectrum, SimulatedSpectrum
from base_model.figures import Figure

#%%
class Simulation:
    """Basic class for simulating a spectrum from input spectra."""

    def __init__(self, input_spectra):
        """
        Initialize the input spectra and an empty SimulatedSpectrum.

        The x-range for the output spectrum is originally the same as
        the first input spectrum.

        The main methods for simulation are:
        - Linear combination of the input spectra
        - changes to the resolution, S/N ratio and x-axis of a spectrum
          as well as simulation of gas phase scattering
        - Plotting of the input and the simulated spectrum

        Parameters
        ----------
        input_spectra : list
            List of instances of the MeasuredSpectrum class.

        Returns
        -------
        None.

        """
        self.input_spectra = input_spectra
        self.core_spectra = [
            s
            for s in self.input_spectra
            if s.spectrum_type == "core_level"
        ]

        self.auger_spectra = [
            s for s in self.input_spectra if s.spectrum_type == "auger"
        ]

        # Initialize the axes and label to the spectrum loaded first.
        input_spectrum = input_spectra[0]
        while input_spectrum.spectrum_type != "core_level":
            # Only use parameters of core level spectra.
            input_spectrum = next(iter(input_spectra))

        start = input_spectrum.start
        stop = input_spectrum.stop
        step = input_spectrum.step
        label = ""

        self.output_spectrum = SimulatedSpectrum(
            start, stop, step, label
        )

    def combine_linear(self, scaling_params):
        """
        Perform a linear combination of the input spectra.

        Each spectrum is scaled by a parameter in the range of [0,1].
        All scaling parameter have to add up to 1.

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
        sim_spectra = self.core_spectra

        shifted_auger_spectra = self._position_augers_randomly(
            self.output_spectrum.x, self.auger_spectra
        )
        sim_spectra.extend(shifted_auger_spectra)

        # Make sure that the right amount of params is given.
        if len(sim_spectra) < len(scaling_params):
            print(
                "Please supply the correct amount of scaling parameters."
            )
            print("Simulated spectrum was not changed!")

        elif len(sim_spectra) > len(scaling_params):
            print("Please supply enough scaling parameters.")
            print("Simulated spectrum was not changed!")

        else:
            self.output_spectrum.label = {}
            if np.round(sum(scaling_params), decimals=1) == 1.0:
                output_list = []

                for i, sim_spectrum in enumerate(sim_spectra):
                    # Species = List of input spectra names
                    species = list(sim_spectrum.label.keys())[0]
                    concentration = scaling_params[i]

                    intensity = (
                        sim_spectrum.lineshape * scaling_params[i]
                    )
                    output_list.append(intensity)

                    # For each species, the label gets a new key:value
                    # pair of the format species: concentration
                    self.output_spectrum.label[species] = concentration

                # Linear combination
                self.output_spectrum.lineshape = sum(output_list)

            else:
                print("Scaling parameters have to sum to 1!")
                print("Simulated spectrum was not changed!")

    def _position_augers_randomly(
        self, x, auger_spectra, shift_x=None,
    ):
        """
        Randomly position auger spectra on x.

        Parameters
        ----------
        auger_spectra : list
            A list of MeasuredSpectrum objects of
            spectrum_type "auger".
        shift_x : float, optional
            Integer describing by how many eV the spectrum shall
            be shifted. If None, then the spectrum is shifted by
            a random amount. The default is None.
        max_shift : float, optional
            Max . The default is 5.

        Returns
        -------
        spectrum : MeasuredSpectrum
            The auger spectrum shifted horizontally.

        """
        start = np.min(x)
        stop = np.max(x)

        x1 = np.roll(x, -1)
        diff = np.abs(np.subtract(x, x1))
        step = np.round(np.min(diff[diff != 0]), 3)
        x = x[diff != 0]

        max_shift = int((np.max(x) - np.mean(x)))

        shift_range = np.arange(-max_shift, max_shift, step)
        if not shift_x:
            # Shift by random amount
            r = np.round(
                np.random.randint(0, len(shift_range)), decimals=2
            )
            shift_x = np.round(shift_range[r], 3)

        if -step < shift_x < step:
            shift_x = 0

        shifted_auger_spectra = []

        for auger_spectrum in auger_spectra:
            shifted_auger_spectrum = SimulatedSpectrum(
                start, stop, step, auger_spectrum.label
            )

            # Position auger spectrum in the middle of the window
            m = int(
                np.where(auger_spectrum.x == np.mean(auger_spectrum.x))[
                    0
                ]
            )
            n = int(np.where(x == np.mean(x))[0])
            if n > m:
                shifted_auger_spectrum.lineshape[
                    n - m : n + m
                ] = auger_spectrum.lineshape[: 2 * m]
                shifted_auger_spectrum.lineshape[
                    : n - m
                ] = auger_spectrum.lineshape[0]
                shifted_auger_spectrum.lineshape[
                    n + m :
                ] = auger_spectrum.lineshape[2 * m]
            elif n < m:
                shifted_auger_spectrum.lineshape = auger_spectrum.lineshape[
                    m - n : n + m + 1
                ]

            shifted_auger_spectrum.shift_horizontal(shift_x)
            shifted_auger_spectra.append(shifted_auger_spectrum)

        # print(f"Auger spectra horizontally shifted by {shift_x} eV.")
        return shifted_auger_spectra

    def change_spectrum(self, spectrum=None, **kwargs):
        """
        Simulate artificial changes on a SimulatedSpectrum object.

        Parameters
        ----------
        spectrum : Spectrum, optional
            A Spectrum object can be supplied if one wants to change a
            single input spectrum and not change a spectrum that was
            already created using a linear combination.
            If spectrum == None,then the current output spectrum is
            changed. The default is None.
        **kwargs :
            resolution: int
                To perform a convolution of the spectrum with a
                Gaussian with FWHM = resolution/mean(x) where x is the
                x-axis of the spectrum.
            signal_to_noise: int
                To add poisson-distributed noise at to the spectrum.
                Signal-to-noise describes the S/N ratio of the
                resulting spectrum.
            shift_x: int
                To shift the spectrum by some eV.
            scatterer: dict
                To simulate scattering in a scattering medium defined
                in the dictionary of the format {'label' : str,
                                                 'distance' : float,
                                                 'pressure' : float}.
                'label' is the name of the scatterer.
                 Allowed values: 'default', 'H2', 'He', 'O2', 'N2.'

        Returns
        -------
        None.

        """
        if spectrum is not None:
            # The step width is defined by the measured spectrum.
            # The output spectrum needs to have its step widths
            # redefined.
            self.output_spectrum.lineshape = spectrum.lineshape
            start = spectrum.start
            stop = spectrum.stop
            step = spectrum.step
            self.label = spectrum.label
            self.output_spectrum.x = np.flip(
                np.arange(start, stop + step, step)
            )
        else:
            pass

        if "fwhm" in kwargs.keys():
            self.output_spectrum.resolution = kwargs["fwhm"]
            self.output_spectrum.change_resolution(kwargs["fwhm"])

        if "shift_x" in kwargs.keys():
            self.output_spectrum.shift_x = kwargs["shift_x"]
            self.output_spectrum.shift_horizontal(kwargs["shift_x"])

        if "signal_to_noise" in kwargs.keys():
            self.output_spectrum.signal_to_noise = kwargs[
                "signal_to_noise"
            ]
            self.output_spectrum.add_noise(kwargs["signal_to_noise"])

        if "scatterer" in kwargs.keys():
            scatter_dict = kwargs["scatterer"]
            self.output_spectrum.scatterer = scatter_dict["label"]
            self.output_spectrum.distance = scatter_dict["distance"]
            self.output_spectrum.pressure = scatter_dict["pressure"]

            self.output_spectrum.scatter_in_gas(
                scatter_dict["label"],
                scatter_dict["distance"],
                scatter_dict["pressure"],
            )

    def plot_simulation(self, plot_inputs=False):
        """
        Create Figure objects for the output spectrum.

        Optionally, the input spectra can also be plotted.

        Parameters
        ----------
        plot_inputs : bool, optional
            If plot_inputs, the input spectra are also plotted.
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
                fig_input = Figure(x, y, title=title)
                figs_input.append(fig_input)

        Figure(
            self.output_spectrum.x,
            self.output_spectrum.lineshape,
            title=self.output_spectrum.spectrum_type,
        )


#%%
if __name__ == "__main__":
    datapath = (
        os.path.dirname(os.path.abspath(__file__)).partition(
            "simulation"
        )[0]
        + "data\\references\\NiCoFe"
    )

    filenames = [
        "Co2p_Co_metal",
        "Co2p_CoO",
        "Co2p_Co3O4",
        "NiLMM_Ni_metal",
        "NiLMM_NiO",
        "CoLMM_CoO",
    ]
    input_spectra = []
    for filename in filenames:
        filepath = datapath + "\\" + filename + ".txt"
        input_spectra += [MeasuredSpectrum(filepath)]

    sim = Simulation(input_spectra)

    sim.combine_linear(scaling_params=[0.4, 0.1, 0.1, 0.1, 0.2, 0.1])
    # =============================================================================
    #     sim.change_spectrum(
    #         shift_x=5,
    #         signal_to_noise=20,
    #         fwhm=200,
    #         scatterer={"label": "O2", "distance": 0.2, "pressure": 1.0},
    #     )
    # =============================================================================
    print(
        "Linear combination parameters: "
        + str(sim.output_spectrum.label)
    )
    sim.plot_simulation(plot_inputs=False)
