# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:39:34 2020.

@author: pielsticker

Basic definition of classes and peak shapes for simulating spectra.
"""

import numpy as np
import os
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d


from .converters.data_converter import DataConverter

try:
    from .peaks import Gauss, Lorentz, Voigt, VacuumExcitation, Tougaard
except ImportError as e:
    if (
        str(e)
        == "attempted relative import with no known parent package"
    ):
        pass
    else:
        raise

#%% Basic classes.
def safe_arange_with_edges(start, stop, step):
    """
    In order to avoid float point errors in the division by step.

    Parameters
    ----------
    start : float
        Smallest value.
    stop : float
        Biggest value.
    step : float
        Step size between points.

    Returns
    -------
    ndarray
        1D array with values in the interval (start, stop),
        incremented by step.

    """
    return step * np.arange(start / step, (stop + step) / step)


def _resample_array(y, x0, x1):
    """
    Resample an array (y) which has the same initial spacing
    of another array(x0), based on the spacing of a new
    array(x1).

    Parameters
    ----------
    y : array
        Lineshape array.
    x0 : array
        x array with old spacing.
    x1 : array
        x array with new spacing.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fn = interp1d(x0, y, axis=0, fill_value="extrapolate")
    return fn(x1)


class Spectrum:
    """Basic class for a spectrum."""

    def __init__(self, start, stop, step, label):
        """
        Initialize numpy arrays for the x values and lineshape.

        The x values are defined using the start, stop,
        and step values.

        Parameters
        ----------
        start : float
            Smallest x value.
        stop : float
            Biggest x value.
        step : float
            Step size between points.
        label : str
            Species label of the spectrum.

        Returns
        -------
        None.

        """
        self.start = start
        self.stop = stop
        self.step = step
        self.x = np.flip(
            safe_arange_with_edges(self.start, self.stop, self.step)
        )
        self.clear_lineshape()
        self.label = label
        self.spectrum_type = None

    def clear_lineshape(self):
        """
        Set the lineshape to an array of all zeros.

        The shape depends on the x axis of the spectrum.

        Returns
        -------
        None.

        """
        self.lineshape = np.zeros(
            int(
                np.round(
                    (self.stop - self.start + self.step) / self.step,
                    decimals=1,
                )
            )
        )
        self.x = np.flip(
            safe_arange_with_edges(self.start, self.stop, self.step)
        )

    def normalize(self):
        """
        Normalize the sum of the lineshape to 1.

        Returns
        -------
        None.

        """
        if np.sum(self.lineshape) != 0:
            self.lineshape = self.lineshape / np.nansum(self.lineshape)

    def update_range(self):
        """
        Update the x axis of the spectrum.

        Can be used when the start, stop, or stop values were udpated.

        Returns
        -------
        None.

        """
        self.x = np.flip(
            safe_arange_with_edges(self.start, self.stop, self.step)
        )

    def resample(self, start, stop, step):
        """
        Resample the x and lineshape arrays.

        Parameters
        ----------
        start : int
            New start value for the x array.
        stop : int
            New stop value for the x array.
        step : int
            New step value for the x array.

        Returns
        -------
        None

        """
        new_x = np.flip(safe_arange_with_edges(start, stop, step))

        self.lineshape = _resample_array(self.lineshape, self.x, new_x)

        self.start = start
        self.stop = stop
        self.step = step
        self.update_range()


class MeasuredSpectrum(Spectrum):
    """Class for loading a measured spectrum."""

    def __init__(self, filepath):
        """
        Load the data into a Spectrum object.

        The step size is automatically determined from the last data
        points.

        Parameters
        ----------
        filepath : str
            Filepath of the .txt or .vms file with the data in the
            format of x and y values.

        Returns
        -------
        None.

        """
        self.filepath = filepath
        data = self.load(filepath)[0]

        x = np.array(data["data"]["x"])
        x1 = np.roll(x, -1)
        diff = np.abs(np.subtract(x, x1))
        self.step = np.round(np.min(diff[diff != 0]), 3)
        x = x[diff != 0]
        y = np.array(data["data"]["y0"])[diff != 0]

        self.start = np.round(np.min(x), 3)
        self.stop = np.round(np.max(x), 3)
        species = data["spectrum_type"] + " " + data["group_name"]
        if not hasattr(self, "label"):
            self.label = {species: 1.0}

        super(MeasuredSpectrum, self).__init__(
            self.start, self.stop, self.step, self.label
        )
        self.x = np.round(x, 3)
        self.lineshape = y

        self.spectrum_type = self._distinguish_core_auger(species)

    def load(self, filepath):
        """
        Load the data from a file.

        Can be either VAMAS or TXT.

        Parameters
        ----------
        filepath : str
            The file should be a .vms or .txt file.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.converter = DataConverter()
        self.converter.load(filepath)
        if filepath.rsplit(".")[-1] == "vms":
            for data in self.converter.data:
                if data["settings"]["y_units"] == "counts":
                    y = np.array(data["data"]["y0"])
                    # dwell = data["settings"]["dwell_time"]
                    # scans = data["scans"]
                    # y = y / dwell / scans
                    data["data"]["y0"] = list(y)
                    data["settings"]["y_units"] = "counts_per_second"
                if data["settings"]["x_units"] == "kinetic energy":
                    data["settings"]["x_units"] = "binding energy"
                    excitation_energy = data["settings"][
                        "excitation_energy"
                    ]
                    data["data"]["x"] = [
                        excitation_energy - x for x in data["data"]["x"]
                    ]

        return self.converter.data

    def write(self, output_folder, new_filename):
        """
        Write the spectrum to a new file.

        Parameters
        ----------
        output_folder : str
            Folder path to store the new reference spectrum.
        new_filename : str
            Filename of the new file.
            Can have suffix .txt or .vms.

        Returns
        -------
        None.

        """
        if new_filename.rsplit(".")[-1] == "vms":
            if self.filepath.rsplit(".")[-1] == "txt":
                raise TypeError("Cannot write a TXT spectrum to VAMAS.")

            self.converter.data[0]["settings"][
                "nr_values"
            ] = self.x.shape[0]

            self.converter.data[0]["data"]["y0"] = self.lineshape
            self.converter.data[0]["data"]["y1"] = _resample_array(
                self.converter.data[0]["data"]["y1"],
                self.converter.data[0]["data"]["x"],
                self.x,
            )

            if (
                self.converter.data[0]["settings"]["x_units"]
                == "binding energy"
            ):
                self.converter.data[0]["settings"][
                    "x_units"
                ] = "kinetic energy"
                excitation_energy = self.converter.data[0]["settings"][
                    "excitation_energy"
                ]
                self.converter.data[0]["data"]["x"] = [
                    np.round(excitation_energy - x, 3) for x in self.x
                ]
        else:
            self.converter.data[0]["data"]["x"] = self.x

        filepath_new = os.path.join(output_folder, new_filename)
        self.converter.write(filepath_new)

        print(f"Spectrum written to {new_filename}")

    def _distinguish_core_auger(self, label):
        """
        Check if the spectrum is a core-level or Auger spectrum.

        Parameters
        ----------
        label : str
            Label of a measured spectrum.

        Returns
        -------
        str
            "auger" or "core_level".

        """
        core_levels = [
            "1s",
            "2s",
            "2p",
            "3s",
            "3p",
            "3d",
            "4s",
            "4p",
            "4d",
            "4f",
            "5s",
            "5p",
            "5d",
            "5f",
            "5g",
            "VB",
            "Survey",
            "Fermi level",
        ]

        if any(core_level in label for core_level in core_levels):
            return "core_level"
        return "auger"


class FittedSpectrum(MeasuredSpectrum):
    """Class for loading and resizing a fitted spectrum."""

    def __init__(self, filepath):
        super(FittedSpectrum, self).__init__(filepath)

    def load(self, filepath):
        """
        Overwrite load method from the MeasuredSpectrum class.

        This is done to accomodate header of CasaXPS export and
        associate the spectrum with a number.

        Parameters
        ----------
        filepath: str
            Filepath of the .txt file with the data in the format of
            x and y values.

        Returns
        -------
        label : dict
            Name of the spectrum given in the first row of the xy file.
        data : ndarray
            2D array with the x and y values from the file.

        """
        lines = []
        with open(filepath, "r") as file:
            for line in file.readlines():
                lines += [line]
        # This takes the species given in the first line
        self.label = (
            str(lines[0]).split(" ", maxsplit=2)[2].split(":")[0]
        )
        species = str(lines[0]).split(":")[-1].split("\n")[0]
        self.number = int(
            str(lines[0]).split(" ", maxsplit=2)[1].split(":")[1]
        )
        lines = [[float(i) for i in line.split()] for line in lines[8:]]
        xy_data = np.array(lines)[:, 2:]

        data = [
            {
                "data": {
                    "x": list(xy_data[:, 0]),
                    "y0": list(xy_data[:, 1]),
                },
                "spectrum_type": species,
                "group_name": "mixed",
            }
        ]
        return data


class SyntheticSpectrum(Spectrum):
    """Class for simulating a SyntheticSpectrum."""

    def __init__(self, start, stop, step, label):
        """
        Initialize an x array using the start, stop, and step values.

        Parameters
        ----------
        start : float
            Smallest x value.
        stop : float
            Biggest x value.
        step : float
            Step size of the x array.
        label : str
            Species label of the spectrum.

        Returns
        -------
        None.

        """
        super(SyntheticSpectrum, self).__init__(
            start, stop, step, label
        )
        self.spectrum_type = "synthetic"
        self.components = []

    def build_line(self):
        """
        Build the lineshape.

        The lineshape is build by calling the function method on each
        of the components.

        Returns
        -------
        None.

        """
        self.clear_lineshape()
        for component in self.components:
            y = np.array([component.function(x) for x in self.x])
            self.lineshape = np.add(self.lineshape, y)

    def add_component(self, component, rebuild=True):
        """
        Adding a Peak component to the spectrum.

        Parameters
        ----------
        component : Peak
            A peak object that needs to have a method 'function'.
        rebuild : bool, optional
            If rebuild, the lineshape is rebuild including the
            new component. The default is True.

        Returns
        -------
        None.

        """
        self.components += [component]
        if rebuild:
            self.rebuild()

    def remove_component(self, comp_idx):
        """
        Remove a specific component and rebuild the lineshape.

        Parameters
        ----------
        comp_idx : int
            Index of the component in the self.components list.

        Returns
        -------
        None.

        """
        del self.components[comp_idx]
        self.rebuild()

    def rebuild(self):
        """
        Update the x values and then rebuild the lineshape.

        Returns
        -------
        None.

        """
        self.update_range()
        self.build_line()


class SimulatedSpectrum(Spectrum):
    """Class for simulating a spectrum."""

    def __init__(self, start, stop, step, label):
        """
        Initialize an x and lineshape array.

        The simulated spectrum is created by changes to the resolution,
        S/N ratio and x-axis of a spectrum as well as by simulation of
        gas phase scattering. At first, set all change parameters to
        None.

        Parameters
        ----------
        start : float
            Smallest x value.
        stop : float
            Biggest x value.
        step : float
            Step size of the x array.
        label : str
            Species label of the spectrum.

        Returns
        -------
        None.

        """
        super(SimulatedSpectrum, self).__init__(
            start, stop, step, label
        )
        self.spectrum_type = "simulated"
        self.shift_x = None
        self.signal_to_noise = None
        self.resolution = None
        self.scatterer = None
        self.pressure = None
        self.distance = None

    def shift_horizontal(self, shift_x):
        """
        Shift the output lineshape by some eV.

        Parameters
        ----------
        shift_x : int
            shift_x is in eV.
            shift_x has to be between -8 and 8 to be valid

        Returns
        -------
        None.

        """
        if shift_x is None:
            pass
        else:
            # scale the shift by the step size
            shift = int(np.round(shift_x / self.step, 1))

            begin_value = self.lineshape[0]
            end_value = self.lineshape[-1]

            # Edge handling by concatenating the first/last value
            # so that the lineshape shape is conserved.
            if shift_x < 0:
                self.lineshape = np.concatenate(
                    (
                        np.full(-shift, begin_value),
                        self.lineshape[:shift],
                    )
                )

            elif shift_x > 0:
                self.lineshape = np.concatenate(
                    (self.lineshape[shift:], np.full(shift, end_value))
                )
            else:
                pass

            self.shift_x = shift_x

    def add_noise(self, signal_to_noise):
        """
        Add noise from a Poisson distribution to the lineshape.

        Parameters
        ----------
        signal_to_noise : int
            Integer value for the amount of noise to be added.

        Returns
        -------
        None.
        """
        if signal_to_noise == 0 or signal_to_noise is None:
            pass
        else:
            intensity_max = np.max(self.lineshape)
            intensity_min = np.min(self.lineshape)

            intensity_diff = intensity_max - intensity_min
            noise = intensity_diff / signal_to_noise * 10

            # A poisson distributed noise is multplied by the noise
            # factor and added to the lineshape.
            lamb = 1000
            poisson_noise = (
                noise
                * np.random.poisson(lamb, self.lineshape.shape)
                / lamb
            )

            self.lineshape = self.lineshape + poisson_noise

    def change_resolution(self, resolution):
        """
        Apply Gaussian instrumental broadening.

        This methdod broadens a spectrum assuming a Gaussian kernel.
        The width of the kernel is determined by the resolution.
        In particular, the function will determine the mean wavelength
        and set the Full Width at Half Maximum (FWHM) of the Gaussian
        to (mean wavelength)/resolution.

        Parameters
        ----------
        resolution : int
            The spectral resolution.

        Returns
        -------
        None.

        """
        if resolution == 0 or resolution is None:
            fwhm = resolution
        else:
            fwhm = np.mean(self.x) / float(resolution)
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # To preserve the position of spectral lines, the
            # broadening function must be centered at
            # N//2 - (1-N%2) = N//2 + N%2 - 1.
            len_x = len(self.x)
            step = self.step
            gauss_x = (
                np.arange(len_x, dtype=np.int)
                - sum(divmod(len_x, 2))
                + 1
            ) * step

            # The broadening spectrum is a synthetic spectrum.
            broadening_spectrum = SyntheticSpectrum(
                gauss_x[0], gauss_x[-1], step, label="Gauss"
            )
            broadening_spectrum.add_component(
                Gauss(position=0, width=sigma, intensity=1)
            )

            # This assures that the edges are handled correctly.
            len_y = len(self.lineshape)
            y = np.concatenate(
                (
                    np.ones(len_y) * self.lineshape[0],
                    self.lineshape,
                    np.ones(len_y) * self.lineshape[-1],
                )
            )

            # This performs the convolution of the initial lineshape
            # with the Gaussian kernel.
            # Note: The convolution is performed in the Fourier space.
            result = fftconvolve(
                y, broadening_spectrum.lineshape, mode="same"
            )
            result = result[len_y:-len_y]

            self.lineshape = result
        self.fwhm = fwhm

    def scatter_in_gas(self, label="He", distance=0.8, pressure=1.0):
        """
        Simulate scattering through a gas phase/thin film.

        First the convolved spectra are dotted with the
        factors vector to scale all the inelastic scatterd spectra by
        the Poisson and angular factors. Then the elastically
        scattered and non-scattered spectra are scaled by their
        respective factors.

        Parameters
        ----------
        label : str, optional
            Label of the scatterer. Can be 'He','H2','N2' or 'default'.
            The default is 'He'.
        distance : float, optional
            Distance (in mm) the electrons travel in the gas.
            The default is 0.8.
        pressure : float, optional
            Pressure of the scattering medium in mbar.
            The default is 1.0.

        Returns
        -------
        None.

        """
        if label in ["He", "H2", "N2", "O2"]:
            # Make sure that the ScatteringMedium class is imported.
            from .scatterers import ScatteringMedium

            medium = ScatteringMedium(label)
            # Loss function and lineshape need to have the same step
            # size.
            medium.scatterer.loss_function.step = self.step
            medium.pressure = pressure
            medium.distance = distance
            # Calculate ScatteringMedium attributes based on the
            # inputs.
            medium.convert_distance()
            medium.calc_density()

            # Build the loss function using the parameters in a
            # json file.
            input_datapath = (
                os.path.dirname(os.path.abspath(__file__)).partition(
                    "simulation"
                )[0]
                + "\\data\\scatterers.json"
            )
            medium.scatterer.build_loss_from_json(input_datapath)
            loss_fn = medium.scatterer.loss_function
            loss_lineshape = loss_fn.lineshape

            if np.sum(loss_lineshape) != 0:
                # Normalize the lineshape of the loss function.
                loss_lineshape /= np.sum(loss_lineshape)

            y = self.lineshape

            # move the minimum of the lineshape to 0.
            min_value = np.min(y)
            y -= min_value

            loss = loss_lineshape

            # Pad the imput spectrum with zeros so that it has the same
            # dimensions as the loss function.
            input_spec_padded = np.pad(
                y,
                (loss.shape[0] - y.shape[0], 0),
                "constant",
                constant_values=0,
            )

            # Take Fourier transform of the input spectrum and the
            # loss function.
            fft_input_spec = np.fft.fft(input_spec_padded)
            fft_loss = np.fft.fft(loss)

            # Calculate the poisson factor for inelastic scattering.
            poisson_factor = (
                medium.distance
                * medium.scatterer.inelastic_xsect
                * medium.density
            )
            norm_factor = medium.scatterer.norm_factor
            total_factor = poisson_factor * norm_factor

            exp_factor = np.exp(-1 * poisson_factor)

            # Convolution in real space = Multiplication in Fourier space.
            fft_total = exp_factor * np.multiply(
                fft_input_spec, np.exp(total_factor * fft_loss)
            )

            # Take the inverse Fourier transform of the convolved spectrum.
            total = np.real(np.fft.ifft(fft_total)[-len(y) :])
            result = total + min_value

            self.lineshape = result

            self.scatterer = label
            self.pressure = pressure
            self.distance = distance

        elif label is None:
            pass
        else:
            print("Please enter a valid scatterer label!")


#%%
if __name__ == "__main__":
    from peaks import Gauss, Lorentz, Voigt, VacuumExcitation, Tougaard

    # label = "NiCoFe\\Ni2pCo2pFe2p_Co_metal"
    label = "NiCoFe\\Fe2p_Fe_metal"
    datapath = (
        os.path.dirname(os.path.abspath(__file__)).partition(
            "simulation"
        )[0]
        + "data\\references"
    )

    filepath = datapath + "\\" + label + ".vms"

    measured_spectrum = MeasuredSpectrum(filepath)

    from figures import Figure

    fig = Figure(
        measured_spectrum.x, measured_spectrum.lineshape, title=label
    )
