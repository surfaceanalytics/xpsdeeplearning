# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:27 2020

@author: pielsticker
"""
import numpy as np
import os
import pandas as pd
import json
from time import time
import matplotlib.pyplot as plt


from base_model.spectra import MeasuredSpectrum
from base_model.figures import Figure
from simulation import Simulation

#%%
class Creator():
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
        default_param_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'default_params.json')
            
        with open(param_filepath, 'r') as param_file:
            self.params = json.load(param_file)
                
        else:
            self.params = params
            
        self.no_of_simulations = self.params['no_of_simulations']
                              
        input_datapath = os.path.join(*[os.path.dirname(
            os.path.abspath(__file__)).partition(
                        'augmentation')[0],
            'data',
            'references'])

        self.input_spectra = []
        for label in self.params['input_filenames']:
            filename = os.path.join(input_datapath, label + '.txt')
            self.input_spectra += [MeasuredSpectrum(filename)]
                
        # No. of parameter = no. of linear parameter + 6
        # (one parameter each for resolution, shift_x, signal_to noise,
        # scatterer, distance, pressure)
        self.no_of_linear_params = len(self.params['input_filenames'])
        no_of_params = self.no_of_linear_params + 6
        
        self.augmentation_matrix = np.zeros((self.no_of_simulations,
                                             no_of_params))
        
        if self.params['single'] == True:
            self.create_matrix(single = True)

        else:
            if self.params['variable_no_of_inputs'] == True:
                self.create_matrix(single = False,
                                   variable_no_of_inputs = True)
            else:
                self.create_matrix(single = False,
                                   variable_no_of_inputs = False)

    
    def create_matrix(self, single = False, variable_no_of_inputs = True):
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
            input spectra used in the linear combination will be randomly
            chosen from the interval (1, No. of input spectra).
            The default is True.

        Returns
        -------
        None.

        """
        self.sim_ranges = {
            'shift_x': (-5,5),
            'noise': (1,25),
            'FWHM': (145,722),
            'scatterers':  {
                '0' : 'He',
                '1' : 'H2', 
                '2' : 'N2',
                '3' : 'O2',
                },
            'pressure': (1,100),
            'distance': (1,10),
            }
        
        for key in self.sim_ranges.keys():
            try:
                self.sim_ranges[key] = self.params['sim_ranges'][key]
            except:
                pass
                    
        for i in range(self.no_of_simulations):
            if single == False:  
                if variable_no_of_inputs:
                    # Randomly choose how many spectra shall be combined
                    no_of_spectra = np.random.randint(
                        1, self.no_of_linear_params+1)
                    r = [np.random.uniform(0.1,1.0) for j \
                         in range(no_of_spectra)]
                    linear_params = [k/sum(r) for k in r]
                    
                    # Don't allow parameters below 0.1.
                    for p in linear_params:
                        if p <= 0.1:
                            linear_params[linear_params.index(p)] = 0.0
    
                    linear_params = [k/sum(linear_params) \
                                     for k in linear_params]
                        
                    # Add zeros if no_of_spectra < no_of_linear_params.
                    for _ in range(self.no_of_linear_params - no_of_spectra):
                        linear_params.append(0.0)
                    
                    # Randomly shuffle so that zeros are equally distributed.    
                    np.random.shuffle(linear_params)
                
                else:
                    # Linear parameters
                    r = [np.random.uniform(0.1,1.0) for j \
                         in range(self.no_of_linear_params)]
                    s = sum(r)
                    linear_params = [k/s for k in r]
                
                    while all(p >= 0.1 for p in linear_params) == False:
                        # sample again if one of the parameters is smaller 
                        # than 0.1.
                        r = [np.random.uniform(0.1,1.0) for j in \
                             range(self.no_of_linear_params)]
                        s = sum(r)
                        linear_params = [k/s for k in r]
                        
                
                self.augmentation_matrix[i,0:self.no_of_linear_params] = \
                    linear_params
                
            else:
                q = np.random.choice(list(range(self.no_of_linear_params)))
                self.augmentation_matrix[i,q] = 1.0
                self.augmentation_matrix[i,:q] = 0
                self.augmentation_matrix[i,q+1:] = 0
            
            
            # FWHM
            if self.params['broaden'] != False:               
                self.augmentation_matrix[i,-6] = np.random.randint(
                    self.sim_ranges['FWHM'][0],
                    self.sim_ranges['FWHM'][1])
            else:
                self.augmentation_matrix[i,-6] = 0
            
            # shift_x
            if self.params['shift_x'] != False:               
                shift_range = np.arange(
                    self.sim_ranges['shift_x'][0],
                    self.sim_ranges['shift_x'][1],
                    self.input_spectra[0].step)
                r = np.round(np.random.randint(0,len(shift_range)),
                             decimals = 2)
                if -self.input_spectra[0].step < \
                    shift_range[r] < self.input_spectra[0].step:
                        shift_range[r] = 0
                
                self.augmentation_matrix[i,-5] = shift_range[r]
                
            else:
                self.augmentation_matrix[i,-5] = 0
                        
            # Signal-to-noise
            if self.params['noise'] != False:
                self.augmentation_matrix[i,-4] = np.random.randint(
                    self.sim_ranges['noise'][0]*1000,
                    self.sim_ranges['noise'][1]*1000)/1000
            
            else:
                self.augmentation_matrix[i,-4] = 0
            
            # Scattering
            if self.params['scatter'] != False:
                # Scatterer ID
                self.augmentation_matrix[i,-3] = \
                    np.random.randint(
                        0, len(self.sim_ranges['scatterers'].keys()))
                # Pressure
                self.augmentation_matrix[i,-2] = \
                    np.random.randint(self.sim_ranges['pressure'][0]*10,
                                      self.sim_ranges['pressure'][1]*10)/10
                # Distance
                self.augmentation_matrix[i,-1] = \
                    np.random.randint(self.sim_ranges['distance'][0]*10,
                                      self.sim_ranges['distance'][1]*10)/10
                
            else:
                # Scatterer
                self.augmentation_matrix[:,-3] = None
                # Pressurex
                self.augmentation_matrix[:,-2] = 0
                # Distance
                self.augmentation_matrix[:,-1] = 0
                
    def run(self,
            broaden = True, 
            x_shift = True,
            noise = True,
            scatter = True):
        """
        The artificial spectra and stare createad using the simulation
        class and the augmentation matrix. All data is then stored in 
        a dataframe.

        Parameters
        ----------
        broaden : bool, optional
            If bool, the spectra are artificially broadened.
            The default is True.
        x_shift : bool, optional
            If x_shift, the spectra are shifted horizontally.
            The default is True.
        noise : bool, optional
            If noise, artificial noise is added to the spectra.
            The default is True.
        scatter : bool, optional
            If scatter, scattering through a gas phase is simulated.
            The default is True.

        Returns
        -------
        None.

        """
        if broaden == False:
            self.augmentation_matrix[:,-6] = 0
        if x_shift == False:
            self.augmentation_matrix[:,-5] = 0
        if noise == False:
            self.augmentation_matrix[:,-4] = 0
        if scatter == False:
            self.augmentation_matrix[:,-3] = None
            # Distance
            self.augmentation_matrix[:,-2] = 0 
            # Pressure
            self.augmentation_matrix[:,-1] = 0
            
        dict_list = []
        for i in range(self.no_of_simulations):
            self.sim = Simulation(self.input_spectra)
            scaling_params = \
                self.augmentation_matrix[i][0:self.no_of_linear_params]
            self.sim.combine_linear(scaling_params = scaling_params)  

            fwhm = self.augmentation_matrix[i][-6] 
            shift_x = self.augmentation_matrix[i][-5] 
            signal_to_noise = self.augmentation_matrix[i][-4] 
            scatterer_id = self.augmentation_matrix[i][-3]            
            distance = self.augmentation_matrix[i][-2]
            pressure = self.augmentation_matrix[i][-1]
            
            try:
                # In order to assign a label, the scatterers are encoded
                # by numbers.
                scatterer_label = \
                    self.sim_ranges['scatterers'][str(int(scatterer_id))]
            except ValueError:
                scatterer_label = None
                       
            self.sim.change_spectrum(
                fwhm = fwhm,
                shift_x = shift_x,
                signal_to_noise = signal_to_noise,
                scatterer = {
                    'label': scatterer_label,
                    'distance' : distance,
                    'pressure' : pressure})
            
            d = self._dict_from_one_simulation(self.sim)
            dict_list.append(d)   
            print('Simulation: ' + str(i+1) + '/' +
                  str(self.no_of_simulations))
            
        self.df = pd.DataFrame(dict_list)
        self.reduced_df = self.df[['x', 'y','label']]
        
        print('Number of created spectra: ' + str(self.no_of_simulations))
            
                     
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
        
        d = {'label': spectrum.label,
             'shift_x': spectrum.shift_x,
             'noise': spectrum.signal_to_noise,
             'FWHM': spectrum.fwhm,
             'scatterer' : spectrum.scatterer,
             'distance' : spectrum.distance,
             'pressure' : spectrum.pressure,
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
            
            linear_params_text = ''
            for key in row['label'].keys():
                linear_params_text += str(key) + ": " + \
                    str(np.round(row['label'][key],decimals =2)) + '\n'
            
            params_text = '\n' 
            if (row['FWHM'] != None and row['FWHM'] != 0):
                params_text += 'FHWM: ' + \
                    str(np.round(row['FWHM'], decimals = 2)) + '\n'
            else:
                params_text += 'FHWM: not changed' + '\n'
                
            if (row['shift_x'] != None and row['shift_x'] != 0):            
                params_text += 'X shift: ' + \
                    '{:.3f}'.format(row['shift_x']) + '\n'
            else:
                params_text += 'X shift: none' + '\n'
                
            if (row['noise'] != None and row['noise'] != 0):
                params_text += 'S/N: ' + '{:.1f}'.format(row['noise']) + '\n' 
            else:
                params_text += 'S/N: not changed' + '\n'

            
            scatter_text = '\n' 
            if row['scatterer'] != None:
                scatter_text += ('Scatterer: ' + 
                                 str(row['scatterer']) + '\n')  
                scatter_text += ('Pressure: ' + 
                                 str(row['pressure']) + ' mbar' + '\n')
                scatter_text += ('Distance: ' + 
                                 str(row['distance']) + ' mm' + '\n')
                 
            else:
                scatter_text += 'Scattering: none' + '\n'
            
            fig.ax.text(0.1, 0.5,
                        linear_params_text + params_text + scatter_text,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform = fig.ax.transAxes,
                        fontsize = 7)
            plt.show()


    def to_file(self, filepath, filetype, how = 'full'):
        """
        Create file from the dataframe of simulated spectra.

        Parameters
        ----------
        filepath : str
            Filepath of the output file.
        filetype : str
            Options: 'excel', 'json', 'txt', 'pickle'
        how : str, optional
            if how == 'full':
                All columns of the dataframe are saved.
            if how == 'reduced':
                Only the  columns x, y, and label are saved.
            The default is 'full'.
        Returns
        -------
        None.

        """
        filetypes = ['excel', 'json', 'pickle']        
        
        if filetype not in filetypes:
            print('Saving was not successful. Choose a valid filetype!')
        else:
            print('Data was saved.')
        
        if how == 'full':
                df = self.df
        elif how == 'reduced':
                df = self.reduced_df
                
        self._save_to_file(df, filepath, filetype)

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
        if filetype == 'excel': 
            file = filename + '.xlsx'
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name = filename)
        
        if filetype == 'json':
            file = filename + '.json'
            with open(file, 'w') as json_file:
                df.to_json(json_file, orient = 'records')
            
        if filetype == 'pickle':
            file = filename + '.pkl'
            with open(file, 'wb') as pickle_file:
                df.to_pickle(pickle_file)


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
    runtime = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                              int(minutes),
                                              seconds)
    
    return runtime
    
#%%
if __name__ == "__main__":
    t0 = time()
    #no_of_simulations = 10
    #input_filenames =  ['Pd_metal_narrow','PdO_narrow']
    #input_filenames = ['Fe_metal','FeO','Fe3O4','Fe2O3']
# =============================================================================
#     creator = Creator(no_of_simulations, input_filenames,
#                       single = False,
#                       variable_no_of_inputs = True)
#     creator.run(broaden = True,
#                 x_shift = True,
#                 noise = True,
#                 scatter = False)
# =============================================================================
    param_filepath = r'C:\Users\pielsticker\Simulations\test.json'
    with open(param_filepath, 'r') as param_file:
            params = json.load(param_file)
            
    creator = Creator(params)
    creator.run()
    creator.plot_random(10)
    datafolder = r'C:\Users\pielsticker\Simulations'
    filepath = os.path.join(datafolder, 'Multiple_species_gas_phase_20200902')
    #creator.upload_to_DB(filename, reduced = True)
    #collections = check_db(filename)
    #drop_db_collection(filename)
# =============================================================================
#     creator.to_file(filepath = filepath,
#                     filetype = 'json',
#                     how = 'full')
# =============================================================================
    t1 = time()
    runtime = calculate_runtime(t0,t1)
    print(f'Runtime: {runtime}.')
    del(t0,t1,runtime,filepath)
    #collections = check_db(filename)
        