# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:27 2020

@author: pielsticker
"""
import numpy as np
import os
import json
import pandas as pd
from time import time
import pickle
import matplotlib.pyplot as plt


from base_model import MeasuredSpectrum, Figure
from simulation import Simulation


class Creator():
    def __init__(self, no_of_simulations):
        self.no_of_simulations = no_of_simulations
        
        datapath = os.path.dirname(
            os.path.abspath(__file__)).partition(
                        'augmentation')[0] + '\\data' + '\\measured'
       
        labels = ['Fe_metal','FeO','Fe3O4','Fe2O3']
        
        self.input_spectra = []
        for label in labels:
            filename = datapath + '\\' + label + '.txt'
            self.input_spectra += [MeasuredSpectrum(filename)]
        
        linear_params = []
        for i in range(len(labels)):
            linear_params.append(1.0)
        
        # No. of parameter = no. of linear parameter + 3
        # (one parameter each for resolution, shift_x, signal_to noise
        self.no_of_linear_params = len(linear_params) 
        no_of_params = self.no_of_linear_params + 3
        
        self.augmentation_matrix = np.zeros((self.no_of_simulations,
                                             no_of_params))
        
        self.create_matrix()

    
    def create_matrix(self):
        """
        Creates the numpy array 'augmenttion_matrix' (instance
        variable) that is used to simulate the new spectra.
        augmentation_matrix has the dimensions (n x p), where:
        n: no. of spectra that will be created
        p: number of parameters
            p = no. of input spectra + 3 (resolution,shift,noise)
        The parameters are chosen randomly. For the last three
        parameters, the random numbers are integers drawn from specific
        intervals that create reasonable spectra-

        Returns
        -------
        None.

        """
        
        shift_x_values = list(range(-9,9,1))
        noise_values = list(range(25,100,5))
        resolution_values = list(range(250,1000,50))
        
        for i in range(self.no_of_simulations):
            # Linear parameters
            r = [np.random.uniform(0.1,1.0) for j \
                 in range(self.no_of_linear_params)]
            s = sum(r)
            linear_params = [ k/s for k in r ]

            while all(p >= 0.1 for p in linear_params) == False:
                # sample again if one of the parameters is smaller than
                # 0.1.
                r = [np.random.uniform(0.1,1.0) for j in \
                     range(self.no_of_linear_params)]
                s = sum(r)
                linear_params = [ k/s for k in r ]

            self.augmentation_matrix[i,0:self.no_of_linear_params] = \
                linear_params

            # FWHM
            self.augmentation_matrix[i][-3] = np.random.randint(250,1500)
            # shift_x
            self.augmentation_matrix[i][-2] = np.random.randint(-8,9)
            # Signal-to-noise
            self.augmentation_matrix[i][-1] = np.random.randint(25,100)
  
                
    def run(self):
        """
        The artificial spectra and stare createad using the simulation
        class and the augmentation matrix. All data is then stored in 
        a dataframe.

        Returns
        -------
        None.

        """
        dict_list = []
        for i in range(self.no_of_simulations):
            sim = Simulation(self.input_spectra)
            scaling_params = \
                self.augmentation_matrix[i][0:self.no_of_linear_params]
            sim.combine_linear(scaling_params = scaling_params)  

            fwhm = self.augmentation_matrix[i][-3] 
            shift_x = self.augmentation_matrix[i][-2] 
            signal_to_noise = self.augmentation_matrix[i][-1] 
            
            sim.change_spectrum(fwhm = fwhm,
                                shift_x = shift_x,
                                signal_to_noise = signal_to_noise)
            
            d = self.dict_from_one_simulation(sim)
            dict_list.append(d)   
            print(i, self.no_of_simulations)
            
        self.df = pd.DataFrame(dict_list)
        self.reduced_df = self.df[['x', 'y','label']]
        
        print("Number of created spectra " + str(self.no_of_simulations))
            
                     
    def dict_from_one_simulation(self, sim):
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
        
        d = {'label': spectrum.label,
             'shift_x': spectrum.shift_x,
             'scale_y': spectrum.scale_y,
             'noise': spectrum.signal_to_noise,
             'FWHM': spectrum.fwhm,
             'x': spectrum.x,
             'y': spectrum.lineshape}
        
        return d
    
    def plot_random(self, no_of_spectra):
        if no_of_spectra > self.no_of_simulations:
            # In this case, plot all spectra.
            no_of_spectra = self.no_of_simulations
        else:
            pass
        
        random_numbers = []
        for i in range(no_of_spectra):
            r = np.random.randint(0,self.no_of_simulations)
            while r in random_numbers:
                # prevent repeating figures
                r = np.random.randint(0,self.no_of_simulations)
                
            random_numbers.append(r)
                    
            row = self.df.iloc[r]
            x = row['x']
            y = row['y']
            title = 'Simulated spectrum no. ' + str(r)
            fig = Figure(x, y, title)
            
            fig_text = ""
            for key in row['label'].keys():
                fig_text += str(key) + ": " + \
                    str(np.round(row['label'][key],decimals =2)) + '\n'
            fig_text += '\n'
            fig_text += 'FHWM: ' + \
                str(np.round(row['FWHM'], decimals = 2)) + '\n'
            fig_text += 'X shift: ' + str(int(row['shift_x'])) + '\n'
            fig_text += 'S/N: ' + str(int(row['noise'])) + '\n'            
 
            fig.ax.text(0.1, 0.45,fig_text,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform = fig.ax.transAxes,
                        fontsize = 9)


    def to_file(self, name, filetype, how = 'full', single = False):
        """
        Create file from the dataframe of simulated spectra

        Parameters
        ----------
        name : str
            Filename of the output file.
        filetype : str
            Options: 'excel', 'json', 'txt', 'pickle'
        how : str, optional
            if how == 'full':
                All columns of the dataframe are saved.
            if how == 'reduced':
                Only the  columns x, y, and label are saved.
            The default is 'full'.
        single : bool, optional
            if single:
                For each spectrum, a single file is created.
                The files are labeled by a sequential number.
            else:
                All spectra are put into one file.

        Returns
        -------
        None.

        """
        filetypes = ['excel', 'json', 'pickle']
        datafolder = os.path.dirname(
            os.path.abspath(__file__)).partition(
                        'augmentation')[0] + '\\data' + '\\\\simulated' 
        
        filepath = datafolder + '\\' + name
        
        if filetype not in filetypes:
            print('Saving was not successful. Choose a valid filetype!')
        else:
            print('Data was saved.')
        
        if how == 'full':
                df = self.df
        elif how == 'reduced':
                df = self.reduced_df
                
        if single == False:
            self._save_to_file(df, filepath, filetype)
        else:
            filenumber = 0 
            test_data = []
            for i in range(self.no_of_simulations):
                number = name + str(i) 
                filepath = datafolder + '\\' + number
                self._save_to_file(df.iloc[i], filepath, filetype)
                filenumber +=1

#                # Test if saving to json worked for single = True.
#                with open(filename + ".json") as json_file:
#                     test = json.load(json_file)#[i]
#                test_data.append(test) 
#            self.test_df = pd.DataFrame(test_data)

# =============================================================================
#                 # Test if saving to pickle worked for single = True.
# #                with open(filename + ".json") as json_file:
# #                    test = json.load(json_file)#[i]
# #                test_data.append(test) 
# #            self.test_df = pd.DataFrame(test_data)
# =============================================================================

    def _save_to_file(self, df, filename, filetype):
        if filetype == 'excel': 
            file = filename + '.xlsx'
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer,sheet_name=filename)
        
        if filetype == 'json':
            file = filename + ".json"
            with open(file, 'w') as json_file:
                self.test_df = df.to_json(json_file)
#                # Test if saving worked for single = False.
#                self.test_df = pd.read_json(file) 
            
        if filetype == 'pickle':
            file = filename + ".pkl"
            with open(file, 'wb') as pickle_file:
                df.to_pickle(pickle_file)
#                # Test if saving worked for single = False.
#                self.test_df = pd.read_pickle(pickle_file)
        
            
    
    def upload_to_DB(self,collection_name):
        pass
              

        
def calculate_runtime(start, end):
    """
    Function to calculate the runtime between two points and return a
    string of the format hh:mm:ss:ff.

    Parameters
    ----------
    start : float32
        Start time, generated by start = time().
    end : float32
        Start time, generated by end = time().

    Returns
    -------
    return_string : str
        Returns a string of the format hh:mm:ss:ff.

    """
    time = end - start    
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    runtime = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    
    return runtime
        
                
#%%
if __name__ == "__main__":
    t0 = time()
    no_of_simulations = 5000000
    creator = Creator(no_of_simulations)
    creator.run()
    creator.plot_random(20)
    t1 = time()
    runtime = calculate_runtime(t0,t1)
    print(f'Runtime: {runtime}.')
    del(t0,t1,runtime)
    
    creator.to_file(name = 'sim_5000000_full',
                    filetype = 'json',
                    how = 'full',
                    single = False)
    
# =============================================================================
# # Runtime test
# times = []
# simulations = [1,10,100,1000,5000,10000,150000]
# for i in simulations:
#     t0 = time()
#     no_of_simulations = i
#     creator = Creator(no_of_simulations)
#     #creator.run()
#     t1 = time() 
#     runtime = calculate_runtime(t0,t1)
#     print(f'Runtime for {i} spectra: {runtime}.')
#     times.append(runtime)
# 
# plt.plot(simulations, times)
# =============================================================================
