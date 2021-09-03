# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:27 2020.

@author: pielsticker
"""
import warnings
import numpy as np
import os
import pandas as pd
import json
import datetime
import h5py
from time import time
import matplotlib.pyplot as plt


from base_model.spectra import (
    safe_arange_with_edges,
    MeasuredSpectrum,
)
from base_model.figures import Figure
from sim import Simulation

# %%
class Creator:
    """Class for simulating mixed XPS spectra."""

    def __init__(self, params=None):
        """
        Prepare simulation run.

        Loading the input spectra and creating the empty simulation
        matrix based on the number of input spectra.

        Parameters
        ----------
        no_of_simulations : int
            The number of spectra that will be simulated.
        input_filenames : list
            List of strings that defines the seed files for the
            simulations.
        single : bool, optional
            If single, then only one of the input spectra will be used
            for creating a single spectrum. If not single, a linear
            combination of all spectra will be used.
            The default is True.
         variable_no_of_inputs : bool, optional
            If variable_no_of_inputs and if single, then the number of
            input spectra used in the linear combination will be
            randomly chosen from the interval 
            (1, No. of input spectra).
            The default is True.

        Returns
        -------
        None.

        """

        default_param_filename = "default_params.json"
        default_param_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), default_param_filename
        )

        with open(default_param_filepath, "r") as param_file:
            self.params = json.load(param_file)
            self.params["init_param_filepath"] = default_param_filepath

            self.sim_ranges = self.params["sim_ranges"]

        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        self.params["timestamp"] = timestamp

        # Replace the default params with the supplied params
        # if available.
        if params is not None:
            for key in params.keys():
                if key != "sim_ranges":
                    self.params[key] = params[key]
                else:
                    for subkey in params["sim_ranges"].keys():
                        self.sim_ranges[subkey] = params["sim_ranges"][subkey]

        # Print parameter file name.
        print(
            "Parameters were taken from "
            f"{self.params['init_param_filepath']}."
        )
        del self.params["init_param_filepath"]

        self.name = self.params["timestamp"] + "_" + self.params["name"]
        self.no_of_simulations = self.params["no_of_simulations"]

        self.labels = self.params["labels"]
        self.spectra = self.params["spectra"]

        # Warning if core and auger spectra of same species are
        # not scaled together.
        if not self.params["same_auger_core_percentage"]:
            warnings.warn(
                'Auger and core spectra of the same species are not scaled'
                ' together. If you have Auger spectra, you may want to set'
                ' "same_auger_core_percentage" to True!'
            )

        # Load input spectra from all reference sets.
        self.input_spectra = self.load_input_spectra(
            self.params["input_filenames"]
        )

        # No. of parameter = 1 ref_set + no. of linear parameter + 6
        # (one parameter each for resolution, shift_x, signal_to noise,
        # scatterer, distance, pressure)
        self.no_of_linear_params = len(self.spectra)
        no_of_params = 1 + self.no_of_linear_params + 6

        self.simulation_matrix = np.zeros(
            (self.no_of_simulations, no_of_params)
        )

        # Create the parameter matrix for the simulation.
        self.create_matrix(
            single=self.params["single"],
            variable_no_of_inputs=self.params["variable_no_of_inputs"],
            always_auger=self.params["always_auger"],
            always_core=self.params["always_core"],
        )

    def load_input_spectra(self, filenames):
        """
        Load input spectra.

        Load input spectra from all reference sets into DataFrame.
        Will store NaN value if no reference is available.

        Parameters
        ----------
        filenames : dict
            Dictionary of list with filenames to load.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing instances of MeasuredSpectrum.
            Each row contains one set of reference spectra.

        """
        input_spectra = pd.DataFrame(columns=self.spectra)

        input_datapath = os.path.join(
            *[
                os.path.dirname(os.path.abspath(__file__)).partition(
                    "simulation"
                )[0],
                "data",
                "references",
            ]
        )

        input_spectra_list = []
        for set_key, value_list in filenames.items():
            ref_spectra_dict = {}
            for filename in value_list:
                filepath = os.path.join(input_datapath, filename)
                measured_spectrum = MeasuredSpectrum(filepath)
                if self.params["normalize_inputs"]:
                    measured_spectrum.normalize()
                label = next(iter(measured_spectrum.label.keys()))
                ref_spectra_dict[label] = measured_spectrum
            input_spectra_list.append(ref_spectra_dict)

        return pd.concat(
            [input_spectra, pd.DataFrame(input_spectra_list)], join="outer"
        )

    def create_matrix(
        self,
        single=False,
        variable_no_of_inputs=True,
        always_auger=False,
        always_core=True,
    ):
        """
        Create matrix for multiple simulations.

        Creates the numpy array 'simulation_matrix' (instance
        variable) that is used to simulate the new spectra.
        simulation_matrix has the dimensions (n x p), where:
        n: no. of spectra that will be created
        p: number of parameters
            p = no. of input spectra + 3 (resolution,shift,noise)
        The parameters are chosen randomly. For the last three
        parameters, the random numbers are integers drawn from specific
        intervals that create reasonable spectra.

        Parameters
        ----------
        single : bool, optional
            If single, only one input spectrum is taken.
            The default is False.
        variable_no_of_inputs : bool, optional
            If variable_no_of_inputs and if single, then the number of
            input spectra used in the linear combination will be
            randomly chosen from the interval
            (1, No. of input spectra).
            The default is True.
        always_auger : bool, optional
            If always_auger, there will always be at least one
            Auger spectrum in the output spectrum.
            The default is False.
        always_core : bool, optional
            If always_auger, there will always be at least one
            core level spectrum in the output spectrum.
            The default is True.

        Returns
        -------
        None.

        """
        for i in range(self.no_of_simulations):
            key = self.select_reference_set()  # select a set of references
            self.simulation_matrix[i, 0] = int(key)

            self.simulation_matrix[
                i, 1 : self.no_of_linear_params + 1
            ] = self.select_scaling_params(
                key, single, variable_no_of_inputs, always_auger
            )

            self.simulation_matrix[
                i, self.no_of_linear_params + 1 :
            ] = self.select_sim_params(key)

            print(
                "Random parameters: "
                + str(i + 1)
                + "/"
                + str(self.no_of_simulations)
            )

    def select_reference_set(self):
        """
        Randomly select a number for calling one of the reference sets.

        Returns
        -------
        int
            A number between 0 and the total number of input
            reference sets.

        """
        return np.random.randint(0, self.input_spectra.shape[0])

    def select_scaling_params(
        self,
        key,
        single=False,
        variable_no_of_inputs=True,
        always_auger=False,
        always_core=True,
    ):
        """
        Randomly select parameters for linear combination.

        Select scaling parameters for a simulation from one set
        of reference spectra (given by key).

        Parameters
        ----------
        key : int
            Integer number of the reference spectrum set to use.
        single : bool, optional
            If single, only one input spectrum is taken.
            The default is False.
        variable_no_of_inputs : bool, optional
            If variable_no_of_inputs and if single, then the number of
            input spectra used in the linear combination will be
            randomly chosen from the interval
            (1, No. of input spectra).
            The default is True.
        always_auger : bool, optional
            If always_auger, there will always be at least one
            Auger spectrum in the output spectrum.
            The default is False.
        always_core : bool, optional
            If always_auger, there will always be at least one
            core level spectrum in the output spectrum.
            The default is True.

        Returns
        -------
        linear_params : list
            A list of parameters for the linear combination of r
            eference spectra.

        """
        linear_params = [0.0] * self.no_of_linear_params

        # Get input spectra from one set of references.
        inputs = self.input_spectra.iloc[[key]]

        # Select indices where a spectrum is available for this key.
        indices = [
            self.spectra.index(j)
            for j in inputs.columns[inputs.isnull().any() == False].tolist()
        ]
        indices_empty = [
            self.spectra.index(j)
            for j in inputs.columns[inputs.isnull().any()].tolist()
        ]

        # This ensures that always just one Auger region is used.
        auger_spectra = []
        core_spectra = []
        for s in inputs.iloc[0]:
            if str(s) != "nan":
                if s.spectrum_type == "auger":
                    auger_spectra.append(s)
                if s.spectrum_type == "core_level":
                    core_spectra.append(s)
        auger_region = self._select_one_auger_region(auger_spectra)

        selected_auger_spectra = [
            auger_spectrum
            for auger_spectrum in auger_spectra
            if (auger_region in list(auger_spectrum.label.keys())[0])
        ]
        unselected_auger_spectra = [
            auger_spectrum
            for auger_spectrum in auger_spectra
            if (auger_region not in list(auger_spectrum.label.keys())[0])
        ]
        selected_auger_indices = [
            inputs.columns.get_loc(list(s.label.keys())[0])
            for s in selected_auger_spectra
        ]
        unselected_auger_indices = [
            inputs.columns.get_loc(list(s.label.keys())[0])
            for s in unselected_auger_spectra
        ]

        if single:
            # Set one parameter to 1 and others to 0.
            q = np.random.choice(indices)
            linear_params[q] = 1.0
        else:
            if variable_no_of_inputs:
                # Randomly choose how many spectra shall be combined
                no_of_spectra = np.random.randint(1, len(indices) + 1)
                params = [0.0] * no_of_spectra
                while sum(params) == 0.0:
                    params = [
                        np.random.uniform(0.1, 1.0)
                        for j in range(no_of_spectra)
                    ]

                    params = self._normalize_float_list(params)
                    # Don't allow parameters below 0.1.
                    for p in params:
                        if p <= 0.1:
                            params[params.index(p)] = 0.0

                    params = self._normalize_float_list(params)

                    # Add zeros if no_of_spectra < no_of_linear_params.
                    for _ in range(len(indices) - no_of_spectra):
                        params.append(0.0)

            else:
                # Linear parameters
                r = [np.random.uniform(0.1, 1.0) for j in range(len(indices))]
                params = self._normalize_float_list(r)

                while all(p >= 0.1 for p in params) is not False:
                    # sample again if one of the parameters is smaller
                    # than 0.1.
                    r = [
                        np.random.uniform(0.1, 1.0)
                        for j in range(len(indices))
                    ]
                    params = self._normalize_float_list(r)

            # Randomly shuffle so that zeros are equally distributed.
            np.random.shuffle(params)
            # Add linear params at the positions where there
            # are reference spectra available
            param_iter = iter(params)
            for index in indices:
                linear_params[index] = next(param_iter)

            # Making sure that a single spectrum is not moved
            # to the undesired Auger region.
            test_indices = [
                i for i in indices if i not in unselected_auger_indices
            ]
            while all(
                p == 0.0 for p in [linear_params[i] for i in test_indices]
            ):
                np.random.shuffle(params)
                param_iter = iter(params)
                for index in indices:
                    linear_params[index] = next(param_iter)

            # Remove undesired auger spectra
            for index in unselected_auger_indices:
                linear_params[index] = 0.0

            if self.params["same_auger_core_percentage"]:
                # Set percentage for core and auger spectra of the
                # same species to the same value.
                linear_params = self._scale_auger_core_together(
                    linear_params, selected_auger_spectra, core_spectra
                )

            linear_params = self._normalize_float_list(linear_params)

        # If no spectrum is available, set the corresponding
        # linear param to NaN.
        for index in indices_empty:
            linear_params[index] = float("NAN")

        if always_auger:
            # Always use at least one Auger spectrum
            # when available.
            if all(
                p == 0.0
                for p in [linear_params[i] for i in selected_auger_indices]
            ):
                linear_params = self.select_scaling_params(
                    key=key,
                    single=single,
                    variable_no_of_inputs=variable_no_of_inputs,
                    always_auger=always_auger,
                    always_core=always_core,
                )
        if always_core:
            # Always use at least one core level spectrum
            # when available.
            core_level_indices = [
                inputs.columns.get_loc(list(s.label.keys())[0])
                for s in core_spectra
            ]
            if all(
                p == 0.0
                for p in [linear_params[i] for i in core_level_indices]
            ):
                linear_params = self.select_scaling_params(
                    key=key,
                    single=single,
                    variable_no_of_inputs=variable_no_of_inputs,
                    always_auger=always_auger,
                    always_core=always_core,
                )

        return linear_params

    def select_sim_params(self, row):
        """
        Select parameters for one row in the simulation matrix.

        Parameters
        ----------
        row : int
            Row in the simulation matrix to fill.

        Returns
        -------
        sim_params : list
            List of parameters for changing a spectrum using
            various processing steps.

        """
        sim_params = [0.0] * 6

        # FWHM
        sim_params[-6] = self._select_random_fwhm()

        # shift_x
        # Get step from first existing spectrum
        for spectrum in self.input_spectra.iloc[row]:
            try:
                step = spectrum.step
                break
            except AttributeError:
                continue
        sim_params[-5] = self._select_random_shift_x(step)

        # Signal-to-noise
        sim_params[-4] = self._select_random_noise()

        # Scattering
        # Scatterer
        sim_params[-3] = self._select_random_scatterer()
        # Pressurex
        sim_params[-2] = self._select_random_scatter_pressure()
        # Distance
        sim_params[-1] = self._select_random_scatter_distance()

        return sim_params

    def _select_random_fwhm(self):
        if self.params["broaden"] is not False:
            return np.random.randint(
                self.sim_ranges["FWHM"][0], self.sim_ranges["FWHM"][1]
            )
        return 0

    def _select_random_shift_x(self, step):
        if self.params["shift_x"] is not False:
            shift_range = np.arange(
                self.sim_ranges["shift_x"][0],
                self.sim_ranges["shift_x"][1],
                step,
            )
            r = np.round(np.random.randint(0, len(shift_range)), decimals=2)
            if -step < shift_range[r] < step:
                shift_range[r] = 0

            return shift_range[r]
        return 0

    def _select_random_noise(self):
        if self.params["noise"] is not False:
            return (
                np.random.randint(
                    self.sim_ranges["noise"][0] * 1000,
                    self.sim_ranges["noise"][1] * 1000,
                )
                / 1000
            )
        return 0

    def _select_random_scatterer(self):
        if self.params["scatter"] is not False:
            # Scatterer ID
            return np.random.randint(
                0, len(self.sim_ranges["scatterers"].keys())
            )
        return None

    def _select_random_scatter_pressure(self):
        if self.params["scatter"] is not False:
            return (
                np.random.randint(
                    self.sim_ranges["pressure"][0] * 10,
                    self.sim_ranges["pressure"][1] * 10,
                )
                / 10
            )
        return 0

    def _select_random_scatter_distance(self):
        if self.params["scatter"] is not False:
            return (
                np.random.randint(
                    self.sim_ranges["distance"][0] * 100,
                    self.sim_ranges["distance"][1] * 100,
                )
                / 100
            )
        return 0

    def _select_one_auger_region(self, auger_spectra):
        """
        Randomly select one region of Auger spectra.

        Checks for the available Auger spectra and randomly
        selects one emission line.

        Returns
        -------
        str
            Name of the randomly selected Auger region.

        """
        labels = [list(s.label.keys())[0] for s in auger_spectra]
        if not labels:
            return []

        auger_regions = set([label.split(" ", 1)[0] for label in labels])
        auger_regions = list(auger_regions)

        r = np.random.randint(0, len(auger_regions))

        return auger_regions[r]

    def _scale_auger_core_together(
        self, linear_params, selected_auger_spectra, core_spectra
    ):
        """
        Set core/auger percentage of the same species to same value.

        Parameters
        ----------
        linear_params : list
            List of all linear parameters.
        selected_auger_spectra : list
            List of all selected Auger spectra.
        core_spectra : list
            List of all avialble core level spectra.

        Returns
        -------
        linear_params : list
            New list of linear parameters where core and auger spectra
            of the same species have the same percentage.

        """
        auger_labels = [
            list(s.label.keys())[0] for s in selected_auger_spectra
        ]
        auger_phases = [label.split(" ", 1)[1] for label in auger_labels]
        core_labels = [list(s.label.keys())[0] for s in core_spectra]
        core_phases = [label.split(" ", 1)[1] for label in core_labels]
        overlapping_species = [
            phase for phase in auger_phases if phase in core_phases
        ]

        for species in overlapping_species:
            label_auger = [
                label for label in auger_labels
                if species in label][0]
            label_core = [
                label for label in core_labels
                if species in label][0]
            i_auger = self.spectra.index(label_auger)
            i_core = self.spectra.index(label_core)
            max_value = np.max(
                [linear_params[i_auger], linear_params[i_core]]
            )
            linear_params[i_auger] = max_value
            linear_params[i_core] = max_value

        return linear_params

    def _normalize_float_list(self, list_of_floats):
        """
        Normalize a list of float by its sum.

        Parameters
        ----------
        param_list : list
            List of floats.

        Returns
        -------
        list
            Normalized list of floats
            (or original list if all entries are 0.)

        """
        try:
            return [k / sum(list_of_floats) for k in list_of_floats]
        except ZeroDivisionError:
            return list_of_floats

    def run(self):
        """
        Run the simulations.

        The artificial spectra are createad using the Simulation
        class and the simulation matrix. All data is then stored in
        a dataframe.

        Returns
        -------
        None.

        """
        dict_list = []
        for i in range(self.no_of_simulations):
            ref_set_key = int(self.simulation_matrix[i, 0])

            # Only select input spectra and scaling parameter
            # for the references that are avalable.
            sim_input_spectra = [
                spectrum
                for spectrum in self.input_spectra.iloc[ref_set_key].tolist()
                if str(spectrum) != "nan"
            ]

            scaling_params = [
                p
                for p in self.simulation_matrix[i][
                    1 : self.no_of_linear_params + 1
                ]
                if str(p) != "nan"
            ]

            self.sim = Simulation(sim_input_spectra)

            self.sim.combine_linear(scaling_params=scaling_params)

            fwhm = self.simulation_matrix[i][-6]
            shift_x = self.simulation_matrix[i][-5]
            signal_to_noise = self.simulation_matrix[i][-4]
            scatterer_id = self.simulation_matrix[i][-3]
            distance = self.simulation_matrix[i][-2]
            pressure = self.simulation_matrix[i][-1]

            try:
                # In order to assign a label, the scatterers are encoded
                # by numbers.
                scatterer_label = self.sim_ranges["scatterers"][
                    str(int(scatterer_id))
                ]
            except ValueError:
                scatterer_label = None

            self.sim.change_spectrum(
                fwhm=fwhm,
                shift_x=shift_x,
                signal_to_noise=signal_to_noise,
                scatterer={
                    "label": scatterer_label,
                    "distance": distance,
                    "pressure": pressure,
                },
            )

            if self.params["normalize_outputs"]:
                self.sim.output_spectrum.normalize()

            d1 = {"reference_set": ref_set_key}
            d2 = self._dict_from_one_simulation(self.sim)
            d = {**d1, **d2}
            dict_list.append(d)
            print(
                "Simulation: "
                + str(i + 1)
                + "/"
                + str(self.no_of_simulations)
            )

        print("Number of created spectra: " + str(self.no_of_simulations))

        self.df = pd.DataFrame(dict_list)

        if self.params["ensure_same_length"]:
            self.df = self._extend_spectra_in_df(self.df)

        self._prepare_metadata_after_run()

        return self.df

    def _dict_from_one_simulation(self, sim):
        """
        Create a dictionary with data from one simulation event.

        Parameters
        ----------
        sim : Simulation
            The simulation for which the dictionary shall be created.

        Returns
        -------
        d : dict
            Dictionaty containing all simulation data.

        """
        spectrum = sim.output_spectrum

        # Add all percentages of one species together.
        new_label = {}
        out_phases = []
        for key, value in spectrum.label.items():
            phase = key.split(" ", 1)[1]
            if phase not in out_phases:
                new_label[phase] = value
            else:
                new_label[phase] += value
            out_phases.append(phase)

        spectrum.label = new_label

        y = np.reshape(spectrum.lineshape, (spectrum.lineshape.shape[0], -1))

        d = {
            "label": spectrum.label,
            "shift_x": spectrum.shift_x,
            "noise": spectrum.signal_to_noise,
            "FWHM": spectrum.fwhm,
            "scatterer": spectrum.scatterer,
            "distance": spectrum.distance,
            "pressure": spectrum.pressure,
            "x": spectrum.x,
            "y": y,
        }
        for label_value in self.labels:
            if label_value not in d["label"].keys():
                d["label"][label_value] = 0.0

        return d

    def _extend_spectra_in_df(self, df):
        """
        Extend all x and y columns in the dataframe to the same length.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with "x" and "y" columns containing 1D numpy
            arrays.

        Returns
        -------
        df : pd.DataFrame
            The same dataframe as the input, with the arrays in the
            "x" and "y" columns all having the same shape.
        """
        max_length = np.max([y.shape for y in self.df["y"].tolist()])

        data_list = df[["x", "y"]].values.tolist()
        new_spectra = []

        for (x, y) in data_list:
            x_new, y_new = self._extend_xy(x, y, max_length)
            new_data_dict = {"x": x_new, "y": y_new}
            new_spectra.append(new_data_dict)

        df.update(pd.DataFrame(new_spectra))

        return df

    def _extend_xy(self, X0, Y0, new_length):
        """
        Extend two 1D arrays to a new length.

        Parameters
        ----------
        X0 : TYPE
            Regularly spaced 1D array.
        Y0 : TYPE
            1D array of the same size as X0.
        new_length : int
            Length of new array.

        Returns
        -------
        None.

        """
        """


        Returns
        -------
        None.

        """
        def start_stop_step_from_x(x):
            """
            Calculcate start, stop, and step from a regular array.

            Parameters
            ----------
            x : ndarrray
                A numpy array with regular spacing,
                i.e. the same step size between all points.

            Returns
            -------
            start : int
                Minimal value of x.
            stop : int
                Maximal value of x.
            step : float
                Step size between points in x.

            """
            start = np.min(x)
            stop = np.max(x)

            x1 = np.roll(x, -1)
            diff = np.abs(np.subtract(x, x1))
            step = np.round(np.min(diff[diff != 0]), 3)

            return start, stop, step

        len_diff = new_length - X0.shape[0]

        if len_diff > 0.0:
            start0, stop0, step0 = start_stop_step_from_x(X0)
            start = start0 - int(len_diff / 2) * step0
            stop = stop0 + int(len_diff / 2) * step0

            X = np.flip(safe_arange_with_edges(start, stop, step0))
            Y = np.zeros(shape=(X.shape[0], 1))

            Y[: int(len_diff / 2)] = np.mean(Y0[:20])
            Y[int(len_diff / 2) : -int(len_diff / 2)] = Y0
            Y[-int(len_diff / 2) :] = np.mean(Y0[-20])

            return X, Y
        return X0, Y0

    def plot_random(self, no_of_spectra):
        """
        Randomly plot some of the generated spectra.

        Labels and simulation parameters are added as texts.

        Parameters
        ----------
        no_of_spectra : int
            No. of random spectra to be plotted.

        Returns
        -------
        None.

        """
        if no_of_spectra > self.no_of_simulations:
            # In this case, plot all spectra.
            no_of_spectra = self.no_of_simulations

        random_numbers = []
        for i in range(no_of_spectra):
            r = np.random.randint(0, self.no_of_simulations)
            while r in random_numbers:
                # prevent repeating figures
                r = np.random.randint(0, self.no_of_simulations)
            random_numbers.append(r)

            row = self.df.iloc[r]
            x = row["x"]
            y = row["y"]
            title = "Simulated spectrum no. " + str(r)
            fig = Figure(x, y, title)
            spectrum_text = self._write_spectrum_text(row)
            fig.ax.text(
                0.1,
                0.9,
                spectrum_text,
                horizontalalignment="left",
                verticalalignment="top",
                transform=fig.ax.transAxes,
                fontsize=7,
            )
            plt.show()

    def _write_spectrum_text(self, df_row):
        """
        Write the text for a spectrum that is plotted from a df row.

        Parameters
        ----------
        df_row : pd.DataFrame
            Row of a DataFrame created during simulation.

        Returns
        -------
        None

        """
        linear_params_text = ""
        for key in self.labels:
            linear_params_text += (
                str(key)
                + ": "
                + str(np.round(df_row["label"][key], decimals=2))
                + "\n"
            )

        params_text = "\n"
        if df_row["FWHM"] is not None and df_row["FWHM"] != 0:
            params_text += (
                "FHWM: " + str(np.round(df_row["FWHM"], decimals=2)) + "\n"
            )
        else:
            params_text += "FHWM: not changed" + "\n"

        if df_row["shift_x"] is not None and df_row["shift_x"] != 0:
            params_text += (
                "X shift: " + "{:.3f}".format(df_row["shift_x"]) + "\n"
            )
        else:
            params_text += "X shift: none" + "\n"

        if df_row["noise"] is not None and df_row["noise"] != 0:
            params_text += "S/N: " + "{:.1f}".format(df_row["noise"]) + "\n"
        else:
            params_text += "S/N: not changed" + "\n"

        scatter_text = "\n"
        if df_row["scatterer"] is not None:
            scatter_text += "Scatterer: " + str(df_row["scatterer"]) + "\n"
            scatter_text += (
                "Pressure: " + str(df_row["pressure"]) + " mbar" + "\n"
            )
            scatter_text += (
                "Distance: " + str(df_row["distance"]) + " mm" + "\n"
            )

        else:
            scatter_text += "Scattering: none" + "\n"

        return linear_params_text + params_text + scatter_text

    def _prepare_metadata_after_run(self):
        """
        Save the metadata from the simulation run in JSON file.

        Returns
        -------
        None.

        """
        self.params["name"] = self.name
        self.params["energy_range"] = [
            np.min(self.df["x"][0]),
            np.max(self.df["x"][0]),
            np.round(self.df["x"][0][0] - self.df["x"][0][1], 2),
        ]


class FileWriter:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.name = self.params["name"]
        self.labels = self.params["labels"]

    def to_file(self, filetypes, metadata=True):
        """
        Create file from the dataframe of simulated spectra.

        Parameters
        ----------
        filepath : str
            Filepath of the output file.
        filetype : str
            Options: 'excel', 'json', 'txt', 'pickle'
        Returns
        -------
        None.

        """
        datafolder = os.path.join(
            *[self.params["output_datafolder"], self.name]
        )
        try:
            os.makedirs(datafolder)
        except FileExistsError:
            pass

        self.filepath = os.path.join(datafolder, self.name)

        valid_filetypes = ["excel", "json", "pickle", "hdf5"]

        for filetype in filetypes:
            if filetype not in valid_filetypes:
                print("Saving was not successful. Choose a valid filetype!")
            else:
                self._save_to_file(self.df, self.filepath, filetype)
                print(f"Data was saved to {filetype.upper()} file.")

        if metadata:
            self.save_metadata()

    def _save_to_file(self, df, filename, filetype):
        """
        Save a dataframe to a file.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with the simulated data.
        filename : str
            Filename of the new file.
        filetype : str
            If "excel", save the data to an Excel file.
            If "json", save the data to a JSON file.
            If "excel", pickle the data and save it.
            If "hdf5", calle the helper method "prepare_hdf5" and store
            the data in an HDF5 file.


        Returns
        -------
        None.

        """
        if filetype == "excel":
            self.excel_filepath = filename + ".xlsx"
            with pd.ExcelWriter(self.excel_filepath) as writer:
                df.to_excel(writer, sheet_name=filename)

        if filetype == "json":          
            self.json_filepath = filename + ".json"
            
            with open(self.json_filepath, "w") as json_file:
                df.to_json(json_file, orient="records")

        if filetype == "pickle":
            self.pkl_filepath = filename + ".pkl"
            with open(self.pkl_filepath, "wb") as pickle_file:
                df.to_pickle(pickle_file)

        if filetype == "hdf5":
            print("Saving data to HDF5...")
            self.hdf5_filepath = filename + ".h5"

            hdf5_data = self.prepare_hdf5(self.df)

            with h5py.File(self.hdf5_filepath, "w") as hf:
                for key, value in hdf5_data.items():
                    try:
                        hf.create_dataset(
                            key, data=value, compression="gzip", chunks=True
                        )
                    except TypeError:
                        value = np.array(value, dtype=object)
                        string_dt = h5py.special_dtype(vlen=str)
                        hf.create_dataset(
                            key,
                            data=value,
                            dtype=string_dt,
                            compression="gzip",
                            chunks=True,
                        )
                    print("Saved " + key + " to HDF5 file.")

    def prepare_hdf5(self, df):
        """
        Store the DataFrame from a simulation run in a dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the result of a simulation process.

        Returns
        -------
        dict
            Dictionary containing all simulated data.
            Items include X, y, shiftx, noise, FWHM, scatterer,
            distance, and pressure,

        """
        X = []
        y = []
        shiftx = []
        noise = []
        FWHM = []
        scatterer = []
        distance = []
        pressure = []

        energies = df["x"][0]
        for index, row in df.iterrows():
            X_one = row["y"]
            if self.params["eV_window"]:
                from pandas.core.common import SettingWithCopyWarning

                warnings.simplefilter(
                    action="ignore", category=SettingWithCopyWarning
                )
                # Only select a random window of some eV as output.
                step = self.params["energy_range"][-1]
                eV_window = self.params["eV_window"]
                window = int(eV_window / step) + 1
                r = np.random.randint(0, X_one.shape[0] - window)
                X_one = X_one[r : r + window]
                self.df["x"][index] = np.flip(
                    safe_arange_with_edges(0, eV_window, step)
                )
                self.df["y"][index] = X_one
                energies = self.df["x"][index]
            y_one = row["label"]
            shiftx_one = row["shift_x"]
            noise_one = row["noise"]
            FWHM_one = row["FWHM"]
            scatterer_name = row["scatterer"]
            scatterers = {"He": 0, "H2": 1, "N2": 2, "O2": 3}
            try:
                scatterer_one = scatterers[scatterer_name]
            except KeyError:
                scatterer_one = float("NAN")
            distance_one = row["distance"]
            pressure_one = row["pressure"]

            X.append(X_one)
            y.append(y_one)
            shiftx.append(shiftx_one)
            noise.append(noise_one)
            FWHM.append(FWHM_one)
            scatterer.append(scatterer_one)
            distance.append(distance_one)
            pressure.append(pressure_one)

            print(
                "Prepare HDF5 upload: " + str(index) + "/" + str(df.shape[0])
            )

        X = np.array(X, dtype=np.float)
        try:
            X = np.reshape(X, (X.shape[0], X.shape[1], -1))
        except IndexError:
            raise IndexError(
                'Could not concatenate individual spectra because their'
                'sizes are different. Either set "ensure_same_length"'
                'to True or "eV_window" to a finite integer!'
            )

        y = self._one_hot_encode(y)

        shiftx = np.reshape(np.array(shiftx), (-1, 1))
        noise = np.reshape(np.array(noise), (-1, 1))
        FWHM = np.reshape(np.array(FWHM), (-1, 1))
        scatterer = np.reshape(np.array(scatterer), (-1, 1))
        distance = np.reshape(np.array(distance), (-1, 1))
        pressure = np.reshape(np.array(pressure), (-1, 1))

        return {
            "X": X,
            "y": y,
            "shiftx": shiftx,
            "noise": noise,
            "FWHM": FWHM,
            "scatterer": scatterer,
            "distance": distance,
            "pressure": pressure,
            "energies": energies,
            "labels": self.labels,
        }

    def _one_hot_encode(self, y):
        """
        One-hot encode the labels.

        As an example, if the label of a spectrum is Fe metal = 1 and all
        oxides = 0, then the output will be np.array([1,0,0,0],1).

        Parameters
        ----------
        y : list
            List of label strings.

        Returns
        -------
        new_labels : arr
            One-hot encoded labels.

        """
        new_labels = np.zeros((len(y), len(self.labels)))

        for i, d in enumerate(y):
            for species, value in d.items():
                number = self.labels.index(species)
                new_labels[i, number] = value

        return new_labels

    def save_metadata(self):
        """
        Save simulation parameters to JSON file.

        Returns
        -------
        None.

        """
        json_filepath = self.filepath + "_metadata.json"

        with open(json_filepath, "w") as out_file:
            json.dump(self.params, out_file, indent=4)


def calculate_runtime(start, end):
    """
    Calculate the runtime between two points.

    Parameters
    ----------
    start : float
        Start time, generated by start = time().
    end : float
        Start time, generated by end = time().

    Returns
    -------
    runtime : str
        Returns a string of the format hh:mm:ss:ff.

    """
    time = end - start
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    runtime = "{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds
    )

    return runtime


# %%
if __name__ == "__main__":
    init_param_filepath = r"C:\Users\pielsticker\Simulations\init_params_Fe_Co_Ni_core_auger.json"
    with open(init_param_filepath, "r") as param_file:
        params = json.load(param_file)

    t0 = time()
    creator = Creator(params=params)
    df = creator.run()
    creator.plot_random(20)
    # creator.to_file(filetypes = ['hdf5'], metadata=True)
    t1 = time()
    runtime = calculate_runtime(t0, t1)
    print(f"Runtime: {runtime}.")
