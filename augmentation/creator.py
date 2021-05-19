# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:27 2020

@author: pielsticker
"""
import numpy as np
import os
import pandas as pd
import json
import datetime
import h5py
from time import time
import matplotlib.pyplot as plt


from base_model.spectra import MeasuredSpectrum
from base_model.figures import Figure
from simulation import Simulation

#%%
class Creator:
    """
    Class for simulating large amounts of XPS spectra based on a 
    number of input_spectra
    """

    def __init__(self, params=None):
        """
        Loading the input spectra and creating the empty augmentation
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
            input spectra used in the linear combination will be randomly
            chosen from the interval (1, No. of input spectra).
            The default is True.         

        Returns
        -------
        None.

        """
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d")
        self.name =  self.timestamp + "_" + params["run_name"]
        
        default_param_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "default_params.json"
        )

        with open(default_param_filepath, "r") as param_file:
            self.params = json.load(param_file)
            self.sim_ranges = self.params["sim_ranges"]

        # Replace the default params with the supplied params
        # if available.
        if params is not None:
            for key in params.keys():
                if key != "sim_ranges":
                    self.params[key] = params[key]
                else:
                    for subkey in params["sim_ranges"].keys():
                        self.sim_ranges[subkey] = params["sim_ranges"][subkey]

        self.no_of_simulations = self.params["no_of_simulations"]
        self.labels = self.params["labels"]

        # Load input spectra from all reference sets.
        self.input_spectra = self.load_input_spectra(
            self.params["input_filenames"]
        )

        # No. of parameter = 1 ref_set + no. of linear parameter + 6
        # (one parameter each for resolution, shift_x, signal_to noise,
        # scatterer, distance, pressure)
        self.no_of_linear_params = len(self.labels)
        no_of_params = 1 + self.no_of_linear_params + 6

        self.augmentation_matrix = np.zeros(
            (self.no_of_simulations, no_of_params)
        )

        if self.params["single"] is True:
            self.create_matrix(single=True)
        else:
            if self.params["variable_no_of_inputs"] is True:
                self.create_matrix(single=False, variable_no_of_inputs=True)
            else:
                self.create_matrix(single=False, variable_no_of_inputs=False)

    def load_input_spectra(self, filenames):
        """
        Load input spectra from all reference sets into DataFrame.
        Will store NaN value if no reference is available.        

        Parameters
        ----------
        filenames : TYPE
            DESCRIPTION.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing instances of MeasuredSpectrum.
            Each row contains one set of reference spectra.

        """
        input_spectra = pd.DataFrame(columns=self.labels)

        input_datapath = os.path.join(
            *[
                os.path.dirname(os.path.abspath(__file__)).partition(
                    "augmentation"
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
                label = next(iter(measured_spectrum.label.keys()))
                ref_spectra_dict[label] = measured_spectrum

            input_spectra_list.append(ref_spectra_dict)

        return pd.concat(
            [input_spectra, pd.DataFrame(input_spectra_list)], join="outer"
        )

    def create_matrix(self, single=False, variable_no_of_inputs=True):
        """
        Creates the numpy array 'augmentation_matrix' (instance
        variable) that is used to simulate the new spectra.
        augmentation_matrix has the dimensions (n x p), where:
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
            randomly chosen from the interval (1, No. of input spectra).
            The default is True.

        Returns
        -------
        None.

        """
        for i in range(self.no_of_simulations):
            key = self.select_reference_set()  # select a set of references
            self.augmentation_matrix[i, 0] = int(key)

            self.augmentation_matrix[
                i, 1 : self.no_of_linear_params + 1
            ] = self.select_scaling_params(key, single, variable_no_of_inputs)

            self.augmentation_matrix[
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
            A number between 0 and the number of input reference sets.

        """
        return np.random.randint(0, self.input_spectra.shape[0])

    def select_scaling_params(
        self, key, single=False, variable_no_of_inputs=True
    ):
        """
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
            randomly chosen from the interval (1, No. of input spectra).
            The default is True.

        Returns
        -------
        linear_params : list
            A list of parameters for the linear combination of r
            eference spectra.

        """
        linear_params = [0.0] * self.no_of_linear_params

        ### Only select linear params if a reference spectrum is available.
        inputs = self.input_spectra.iloc[[key]]
        indices = [
            self.labels.index(j)
            for j in inputs.columns[inputs.isnull().any() == False].tolist()
        ]
        indices_empty = [
            self.labels.index(j)
            for j in inputs.columns[inputs.isnull().any()].tolist()
        ]

        if not single:
            if variable_no_of_inputs:
                # Randomly choose how many spectra shall be combined
                no_of_spectra = np.random.randint(1, len(indices) + 1)
                r = [
                    np.random.uniform(0.1, 1.0) for j in range(no_of_spectra)
                ]
                params = [k / sum(r) for k in r]

                # Don't allow parameters below 0.1.
                for p in params:
                    if p <= 0.1:
                        params[params.index(p)] = 0.0

                params = [k / sum(params) for k in params]

                # Add zeros if no_of_spectra < no_of_linear_params.
                for _ in range(len(indices) - no_of_spectra):
                    params.append(0.0)

            else:
                # Linear parameters
                r = [np.random.uniform(0.1, 1.0) for j in range(len(indices))]
                s = sum(r)
                params = [k / s for k in r]

                while all(p >= 0.1 for p in params) is not False:
                    # sample again if one of the parameters is smaller
                    # than 0.1.
                    r = [
                        np.random.uniform(0.1, 1.0)
                        for j in range(len(indices))
                    ]
                    s = sum(r)
                    params = [k / s for k in r]

            # Randomly shuffle so that zeros are equally distributed.
            np.random.shuffle(params)

            # Add linear params at the positions where there
            # are reference spectra available.
            param_iter = iter(params)
            for index in indices:
                linear_params[index] = next(param_iter)

        else:
            q = np.random.choice(indices)
            linear_params[q] = 1.0

        # If no spectrum is avalable, set the linear param to NaN.
        for index in indices_empty:
            linear_params[index] = float("NAN")

        return linear_params

    def select_sim_params(self, row):
        """
        Select simulation parameters for one row in the augmentation matrix.

        Parameters
        ----------
        row : int
            Row in the augmentation matrix to fill.

        Returns
        -------
        sim_params : list
            List of parameters for changing a spectrum using 
            various processing steps.

        """

        sim_params = [0.0] * 6
        step = self.input_spectra.iloc[row][0].step

        # FWHM
        if self.params["broaden"] is not False:
            sim_params[-6] = np.random.randint(
                self.sim_ranges["FWHM"][0], self.sim_ranges["FWHM"][1]
            )
        else:
            self.augmentation_matrix[row, -6] = 0

        # shift_x
        if self.params["shift_x"] is not False:
            shift_range = np.arange(
                self.sim_ranges["shift_x"][0],
                self.sim_ranges["shift_x"][1],
                step,
            )
            r = np.round(np.random.randint(0, len(shift_range)), decimals=2)
            if -step < shift_range[r] < step:
                shift_range[r] = 0

            sim_params[-5] = shift_range[r]

        else:
            sim_params[-5] = 0

        # Signal-to-noise
        if self.params["noise"] is not False:
            sim_params[-4] = (
                np.random.randint(
                    self.sim_ranges["noise"][0] * 1000,
                    self.sim_ranges["noise"][1] * 1000,
                )
                / 1000
            )

        else:
            sim_params[-4] = 0

        # Scattering
        if self.params["scatter"] is not False:
            # Scatterer ID
            sim_params[-3] = np.random.randint(
                0, len(self.sim_ranges["scatterers"].keys())
            )
            # Pressure
            sim_params[-2] = (
                np.random.randint(
                    self.sim_ranges["pressure"][0] * 10,
                    self.sim_ranges["pressure"][1] * 10,
                )
                / 10
            )
            # Distance
            sim_params[-1] = (
                np.random.randint(
                    self.sim_ranges["distance"][0] * 100,
                    self.sim_ranges["distance"][1] * 100,
                )
                / 100
            )

        else:
            # Scatterer
            sim_params[-3] = None
            # Pressurex
            sim_params[-2] = 0
            # Distance
            sim_params[-1] = 0

        return sim_params

    def run(self):
        """
        The artificial spectra are createad using the simulation
        class and the augmentation matrix. All data is then stored in 
        a dataframe.        

        Returns
        -------
        None.

        """
        dict_list = []
        for i in range(self.no_of_simulations):
            ref_set_key = int(self.augmentation_matrix[i, 0])

            # Only select input spectra and scaling parameter
            # for the references that are avalable.
            sim_input_spectra = [
                spectrum
                for spectrum in self.input_spectra.iloc[ref_set_key].tolist()
                if str(spectrum) != "nan"
            ]
            scaling_params = [
                p
                for p in self.augmentation_matrix[i][
                    1 : self.no_of_linear_params + 1
                ]
                if str(p) != "nan"
            ]

            self.sim = Simulation(sim_input_spectra)

            self.sim.combine_linear(scaling_params=scaling_params)

            fwhm = self.augmentation_matrix[i][-6]
            shift_x = self.augmentation_matrix[i][-5]
            signal_to_noise = self.augmentation_matrix[i][-4]
            scatterer_id = self.augmentation_matrix[i][-3]
            distance = self.augmentation_matrix[i][-2]
            pressure = self.augmentation_matrix[i][-1]

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

        self.df = pd.DataFrame(dict_list)
        self.reduced_df = self.df[["x", "y", "label"]]

        print("Number of created spectra: " + str(self.no_of_simulations))

    def _dict_from_one_simulation(self, sim):
        """
        Creates a dictionary containing all information from one
        simulation event.

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

        d = {
            "label": spectrum.label,
            "shift_x": spectrum.shift_x,
            "noise": spectrum.signal_to_noise,
            "FWHM": spectrum.fwhm,
            "scatterer": spectrum.scatterer,
            "distance": spectrum.distance,
            "pressure": spectrum.pressure,
            "x": spectrum.x,
            "y": spectrum.lineshape,
        }

        for label_value in self.labels:
            if label_value not in d["label"].keys():
                d["label"][label_value] = 0.0

        return d

    def plot_random(self, no_of_spectra):
        """
        Randomly plots of the generated spetra.
        Labels and augmentation parameters are added as texts.

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
        else:
            pass

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

            linear_params_text = ""
            for key in row["label"].keys():
                linear_params_text += (
                    str(key)
                    + ": "
                    + str(np.round(row["label"][key], decimals=2))
                    + "\n"
                )

            params_text = "\n"
            if row["FWHM"] is not None and row["FWHM"] != 0:
                params_text += (
                    "FHWM: " + str(np.round(row["FWHM"], decimals=2)) + "\n"
                )
            else:
                params_text += "FHWM: not changed" + "\n"

            if row["shift_x"] is not None and row["shift_x"] != 0:
                params_text += (
                    "X shift: " + "{:.3f}".format(row["shift_x"]) + "\n"
                )
            else:
                params_text += "X shift: none" + "\n"

            if row["noise"] is not None and row["noise"] != 0:
                params_text += "S/N: " + "{:.1f}".format(row["noise"]) + "\n"
            else:
                params_text += "S/N: not changed" + "\n"

            scatter_text = "\n"
            if row["scatterer"] is not None:
                scatter_text += "Scatterer: " + str(row["scatterer"]) + "\n"
                scatter_text += (
                    "Pressure: " + str(row["pressure"]) + " mbar" + "\n"
                )
                scatter_text += (
                    "Distance: " + str(row["distance"]) + " mm" + "\n"
                )

            else:
                scatter_text += "Scattering: none" + "\n"

            fig.ax.text(
                0.1,
                0.5,
                linear_params_text + params_text + scatter_text,
                horizontalalignment="left",
                verticalalignment="top",
                transform=fig.ax.transAxes,
                fontsize=7,
            )
            plt.show()

    def to_file(self, filetypes):
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
        filepath = os.path.join(*[self.params["output_datafolder"],
                                  self.name])
                                  
        valid_filetypes = ["excel", "json", "pickle", "hdf5"]

        for filetype in filetypes:
            if filetype not in valid_filetypes:
                print("Saving was not successful. Choose a valid filetype!")
            else:
                self._save_to_file(self.df, filepath, filetype)
                print("Data was saved.")


    def _save_to_file(self, df, filename, filetype):
        """
        Helper method for saving a dataframe to a file.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with the simulated data.
        filename : str
            Filename of the new file.
        filetype : str
            If 'excel', save the data to an Excel file.
            If 'json', save the data to a JSON file.
            If 'excel', pickle the data and save it.

        Returns
        -------
        None.

        """
        if filetype == "excel":
            file = filename + ".xlsx"
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name=filename)

        if filetype == "json":
            file = filename + ".json"
            with open(file, "w") as json_file:
                df.to_json(json_file, orient="records")

        if filetype == "pickle":
            file = filename + ".pkl"
            with open(file, "wb") as pickle_file:
                df.to_pickle(pickle_file)
        
        if filetype == "hdf5":
            self.hdf5_filepath = filename + ".h5"
            
            hdf5_data = self.prepare_hdf5(self.df)
            
            with h5py.File(self.hdf5_filepath, "w") as hf:
                for key, value in hdf5_data.items():
                    try:
                        hf.create_dataset(
                            key, 
                            data=value, 
                            compression="gzip", 
                            chunks=True)
                    except TypeError:
                        value = np.array(value, dtype=object)
                        string_dt = h5py.special_dtype(vlen=str)
                        hf.create_dataset(
                            key,
                            data=value,
                            dtype=string_dt,
                            compression="gzip",
                            chunks=True)
                    print("Saved " + key + " to HDF5 file.")

            
    def prepare_hdf5(self, df):            
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
                "Prepare HDF5 upload: "
                + str(index)
                + "/"
                + str(df.shape[0])
            )
    
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        y = self._one_hot_encode(y)
    
        shiftx = np.reshape(np.array(shiftx), (-1, 1))
        noise = np.reshape(np.array(noise), (-1, 1))
        FWHM = np.reshape(np.array(FWHM), (-1, 1))
        scatterer = np.reshape(np.array(scatterer), (-1, 1))
        distance = np.reshape(np.array(distance), (-1, 1))
        pressure = np.reshape(np.array(pressure), (-1, 1))
        
        return {"X": X,
                "y": y,
                "shiftx": shiftx,
                "noise": noise,
                "FWHM": FWHM,
                "scatterer": scatterer,
                "distance": distance,
                "pressure": pressure,
                "energies": energies,
                "labels": self.labels}
    
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
    
    def save_run_params(self):
        pass
        


def calculate_runtime(start, end):
    """
    Function to calculate the runtime between two points and return a
    string of the format hh:mm:ss:ff.

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


#%%
if __name__ == "__main__":
    runtimes = {}
    
    init_param_filepath = r"C:\Users\pielsticker\Simulations\init_params.json"
    with open(init_param_filepath, "r") as param_file:
        params = json.load(param_file)

    t0 = time()
    creator = Creator(params)
    creator.run()
    creator.plot_random(10)
    creator.to_file(filetypes = ['hdf5'])
    t1 = time()
    runtime = calculate_runtime(t0, t1)
    print(f"Runtime: {runtime}.")
    
    # Test new file.
    t0 = time()
    with h5py.File(creator.hdf5_filepath, "r") as hf:
        size = hf["X"].shape
        X_h5 = hf["X"][:4000, :, :]
        y_h5 = hf["y"][:4000, :]
        shiftx_h5 = hf["shiftx"][:4000, :]
        noise_h5 = hf["noise"][:4000, :]
        fwhm_h5 = hf["FWHM"][:4000, :]
        energies_h5 = hf["energies"][:]
        labels_h5 = [str(label) for label in hf["labels"][:]]
        scatterers_h5 = hf["scatterer"][:4000, :]



