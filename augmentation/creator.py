# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:21:27 2020

@author: pielsticker
"""
import numpy as np
import os
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import pymongo # MongoDB upload


from base_model.spectra import MeasuredSpectrum
from base_model.figures import Figure
from simulation import Simulation
import credentials
from upload_to_db import connect_to_db

#%%
class Creator():
    """
    Class for simulating large amounts of XPS spectra based on a 
    number of input_spectra
    """
    def __init__(self, no_of_simulations, input_filenames, single = False,
                 variable_no_of_inputs = True):
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
        self.no_of_simulations = no_of_simulations
        
        input_datapath = os.path.join(*[os.path.dirname(
            os.path.abspath(__file__)).partition(
                        'augmentation')[0],
            'data',
            'references'])

        self.input_spectra = []
        for label in input_filenames:
            filename = os.path.join(input_datapath, label + '.txt')
            self.input_spectra += [MeasuredSpectrum(filename)]
                
        # No. of parameter = no. of linear parameter + 6
        # (one parameter each for resolution, shift_x, signal_to noise,
        # scatterer, distance, pressure)
        self.no_of_linear_params = len(input_filenames) 
        no_of_params = self.no_of_linear_params + 6
        
        self.augmentation_matrix = np.zeros((self.no_of_simulations,
                                             no_of_params))
        
        if single:
            self.create_matrix(single = True)

        else:
            if variable_no_of_inputs:
                self.create_matrix(single = False,
                                   variable_no_of_inputs = True)
            else:
                self.create_matrix(single = False,
                                   variable_no_of_inputs = False)

    
    def create_matrix(self, single = False, variable_no_of_inputs = True):
        """
        Creates the numpy array 'augmenttion_matrix' (instance
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
        print(variable_no_of_inputs)

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
            self.augmentation_matrix[i,-6] = np.random.randint(145,722)
            
            # shift_x
            test = np.arange(-5,5,0.05)
            r = np.round(np.random.randint(0,len(test)), decimals = 2)
            
            if -0.05 < test[r] < 0.05:
                test[r] = 0
            
            self.augmentation_matrix[i,-5] = test[r]
            
            # Signal-to-noise
            self.augmentation_matrix[i,-4] = np.random.randint(1000,25000)/1000
            
            # Scatterer ID
            self.augmentation_matrix[i,-3] = np.random.randint(0,4)
            # Distance
            self.augmentation_matrix[i,-2] = np.random.randint(1,10)/10 
            # Pressure
            self.augmentation_matrix[i,-1] = np.random.randint(1,100)/10
  
                
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
            # In order to assign a label, the scatterers are encoded
            # by numbers.
            scatterers = {'0' : 'He',
                          '1' : 'H2', 
                          '2' : 'N2', 
                          '3' : 'O2'}
            
            distance = self.augmentation_matrix[i][-2]
            pressure = self.augmentation_matrix[i][-1]
            
            try:
                scatterer_label = scatterers[str(int(scatterer_id))]
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

                
    def upload_to_DB(self, collection_name, reduced = True):
        """
        Upload the simulated data to a collection in the SIALab
        MongoDB. 

        Parameters
        ----------
        collection_name : str, optional
            The name of the new collections.
            The default is None.
        reduced : bool, optional
            If reduced, only the  columns x, y, and label are saved.
            The default is True.

        Returns
        -------
        None

        """
        client = pymongo.MongoClient(credentials.connectionstring)

        db = client[credentials.db_name]
        
        # Decide whether all or only the reduced data shall be saved.
        if reduced:
            df = self.reduced_df
        else:
            df = self.df
        
        # Check for overwriting.
        write = True
        collections = [collection for collection \
                       in db.list_collection_names()]
        
        while collection_name in collections:
            print('\n SIALAb MongoDB: Collection already exists!')
            answer = None
            while answer not in ('yes','no'):
                answer = input('Do you want to overwrite it? (yes/no)')
                if answer == 'yes':
                    write = True
                    print('Collection overwritten.')
                elif answer == 'no':
                    write = False
                    print('Collection not overwritten.')
                    
                    answer2 = None
                    while answer2 not in ('yes', 'no'):
                        answer2 = input(
                            'Do you want to create a new collection? (yes/no)')
                        if answer2 == "yes":
                            write = True
                            collection_name = input(
                                'Please enter a new collection name.') 
                            print('New collection was uploaded.')
                        elif answer2 == 'no':
                            write = False
                            print('Data was not uploaded.')
                    
                elif answer2 == 'no':
                    write = False
                    
                else:
                    print('Please enter yes or no.')
                    
            if answer == 'yes':
                break
            if answer2 == 'no':
                break

        if write:
            #Upload data to DB 
            db[collection_name].delete_many({})
            for i in range(self.no_of_simulations):
                row = df.iloc[i]
                data = row.to_dict()
                data['X'] = list(np.round(data['x'],decimals = 2))
                data['y'] = list(data['y'])
                db[collection_name].insert_one(data)
                print('Upload: ' + str(i+1) + '/' +
                      str(self.no_of_simulations))
        

        
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


def check_db(collection_name):  
    """
    Check which data was uploaded. 

    Parameters
    ----------
    collection_name : str, optional
        Name of the collection for which the data is to be checked.

    Returns
    -------
    c : dict
        All collections in the db.
    all_data : TYPE
        DESCRIPTION.

    """
    client, db = connect_to_db()
    collection = db[collection_name]
    
    all_data = []
    for doc in collection.find():
        data_single = doc
        # Remove MongoDB id
        del data_single['_id']
        data_single['x'] = np.array(data_single['x'])
        data_single['y'] = np.array(data_single['y'])
        all_data.append(data_single)
        
    c = dict((collection,
              [document for document in db.collection.find()])
             for collection in db.list_collection_names())
    client.close()

    return c, all_data      


def drop_db_collection(collection_name):  
    """
    Removes a collection from the SIALab MongoDB.

    Parameters
    ----------
    collection_name : str, optional
        Name of the collection which shall be deleted.

    Returns
    -------
    None.

    """
    client, db = connect_to_db()
    collection = db[collection_name]
    collection.drop()
    client.close()    

             
#%%
if __name__ == "__main__":
    t0 = time()
    no_of_simulations = 10
    input_filenames =  ['Pd_metal_narrow','PdO_narrow']
    creator = Creator(no_of_simulations, input_filenames,
                      single = False,
                      variable_no_of_inputs = False)
    creator.run(broaden = True,
                x_shift = True,
                noise = True,
                scatter = False)
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
