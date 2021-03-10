# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:29:41 2021

@author: pielsticker
"""

import numpy as np
import h5py
from sklearn.utils import shuffle

from .utils import ClassDistribution, SpectraPlot

#%%
class DataHandler:
    def __init__(self,
                 intensity_only=True):
        self.intensity_only = intensity_only
    
    def load_data_preprocess(self, input_filepath, no_of_examples,
                             train_test_split, train_val_split):
        """
        Load the data from an HDF5 file and preprocess it into three 
        datasets:

        Parameters
        ----------
        input_filepath : str
            Filepath of the .
        no_of_examples : int
            Number of samples from the input file.
        train_test_split : float
            Split percentage between train+val and test set.
            Typically ~ 0.2.
        train_val_split : float
            Split percentage between train and val set.
            Typically ~ 0.2.

        Returns
        -------
        self.X_train : ndarray
            Training features. Shape: 3d Numpy array.
        self.X_val : ndarray
            Validation features.
        self.X_test : ndarray
            Test features.
        self.y_train : ndarray. Shape: 2d Numpy array.
            Training labels.
        self.y_val : ndarray
            Validation labels.
        self.y_test : ndarray
            Test labels.
            
        Optionally, the method can also return more information about
        the data set:
            self.aug_values_train,
            self.aug_values_val,
            self.aug_values_test : dicts
                Dictionary containing information about the parameters
                used during the artificical constructuon of the dataset.
                Keys : 'shiftx', 'noise', 'FWHM',
                       'scatterer', 'distance', 'pressure' 
                Split in the same way as the features and labels.
                
             self.names_train,
             self.names_val, 
             self.names_test : ndarrays
                 Arrays of the spectra names associated  with each X,y
                 pairing in the data set. Typical for measured spectra.   
                 Split in the same way as the features and labels.
        """
        self.input_filepath = input_filepath
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.no_of_examples = no_of_examples
        
        
        with h5py.File(input_filepath, 'r') as hf:
            try:
                self.energies = hf['energies'][:]
            except KeyError:
                self.energies = np.flip(np.arange(694, 750.05, 0.05))
                print('The data set did not an energy scale. ' +
                      'Default (Fe) was assumed.')
            try:
                self.labels = [str(label) for label in hf['labels'][:]]
                self.num_classes = len(self.labels)
            except KeyError:
                print('The data set did not contain any labels. ' +
                      'The label list is empty.')
                
            dataset_size = hf['X'].shape[0]
            # Randomly choose a subset of the whole data set.
            r = np.random.randint(0, dataset_size-self.no_of_examples)
            X = hf['X'][r:r+self.no_of_examples, :, :]
            y = hf['y'][r:r+self.no_of_examples, :]
            
            if not self.intensity_only:
                new_energies = np.tile(
                    np.reshape(np.array(self.energies),(-1,1)),
                    (X.shape[0],1,1))
                X = np.dstack((new_energies, X))
            
            # Check if the data set was artificially created.
            if 'shiftx' in hf.keys():
                shift_x = hf['shiftx'][r:r+self.no_of_examples, :]
                noise = hf['noise'][r:r+self.no_of_examples, :]
                fwhm = hf['FWHM'][r:r+self.no_of_examples, :] 
                
                if 'scatterer' in hf.keys():
                    # If the data set was artificially created, check 
                    # if scattering in a gas phase was simulated.
                    scatterer = hf['scatterer'][r:r+self.no_of_examples, :]
                    distance = hf['distance'][r:r+self.no_of_examples, :]
                    pressure = hf['pressure'][r:r+self.no_of_examples, :]
                    
                    self.X, self.y, shift_x, noise, fwhm, \
                        scatterer, distance, pressure = \
                            shuffle(X, y, shift_x, noise, fwhm,
                                    scatterer, distance, pressure) 
                            
                    # Store all parameters of the simulations in a dict
                    aug_values = {'shift_x' : shift_x,
                                  'noise' : noise,
                                  'fwhm' : fwhm,
                                  'scatterer' : scatterer,
                                  'distance' : distance,
                                  'pressure' : pressure}
                    self.aug_values = aug_values 
                    
                else:
                    # Shuffle all arrays together
                    self.X, self.y, shift_x, noise, fwhm = \
                        shuffle(X, y, shift_x, noise, fwhm)
                    aug_values = {'shift_x' : shift_x,
                                  'noise' : noise,
                                  'fwhm' : fwhm}
                    self.aug_values = aug_values 
                    
            # Split into train, val and test sets
                self.X_train, self.X_val, self.X_test, \
                    self.y_train, self.y_val, self.y_test, \
                        self.aug_values_train, self.aug_values_val, \
                            self.aug_values_test = \
                                self._split_test_val_train(
                                    self.X, self.y,
                                    aug_values = self.aug_values)
                
                # Determine the shape of the training features, 
                # needed for model building in Keras. 
                self.input_shape = (self.X_train.shape[1],
                                    self.X_train.shape[2])
            
                return self.X_train, self.X_val, self.X_test, \
                       self.y_train, self.y_val, self.y_test, \
                       self.aug_values_train, self.aug_values_val, \
                       self.aug_values_test
                
            # Check if the data have associated names. Typical for
            # measured spectra.
            elif 'names' in hf.keys():
                names_load_list = [name[0].decode("utf-8") for name
                                   in hf['names'][r:r+self.no_of_examples, :]]
                names = np.reshape(np.array(names_load_list),(-1,1))
                
                # Shuffle all arrays together
                self.X, self.y, self.names = \
                        shuffle(X,y , names)

                # Split into train, val and test sets
                self.X_train, self.X_val, self.X_test, \
                    self.y_train, self.y_val, self.y_test, \
                      self.names_train, self.names_val, \
                          self.names_test = \
                              self._split_test_val_train(
                                  self.X, self.y, names = self.names)

                # Determine the shape of the training features, 
                # needed for model building in Keras. 
                self.input_shape = (self.X_train.shape[1],
                                    self.X_train.shape[2])
                                    
                return self.X_train, self.X_val, self.X_test, \
                       self.y_train, self.y_val, self.y_test, \
                       self.names_train, self.names_val, self.names_test
            
            # If there are neither augmentation values nor names in 
            # the dataset, just load the X and y arrays.
            else:
                # Shuffle X and y together
                self.X, self.y = shuffle(X, y)
                # Split into train, val and test sets
                self.X_train, self.X_val, self.X_test, \
                self.y_train, self.y_val, self.y_test = \
                    self._split_test_val_train(self.X, self.y)
                
                # Determine the shape of the training features, 
                # needed for model building in Keras. 
                self.input_shape = (self.X_train.shape[1],
                                    self.X_train.shape[2])
                
                return self.X_train, self.X_val, self.X_test, \
                       self.y_train, self.y_val, self.y_test
    
    def _split_test_val_train(self, X, y, **kwargs):
        """
        Helper method for splitting multiple numpy arrays two times:
        First, the whole data is split into the train+val and test sets
        according to the attribute self.train_test_split. Secondly.
        the train+val sets are further split into train and val sets 
        according to the attribute self.train_val_split.

        Parameters
        ----------
        X : ndarray
            Features used as inputs for a Keras model. 3d array.
        y : TYPE
            Labels to be learned. 2d array.
        **kwargs : str
            Possible keywords:
                'aug_values', 'names'.

        Returns
        -------
        For each input array, three output arras are returned.
        E.g. for input X, the returns are X_train, X_val, X_test.
        """
        # First split into train+val and test sets
        no_of_train_val = int((1-self.train_test_split) *\
                              X.shape[0])
        
        X_train_val = X[:no_of_train_val,:,:]
        X_test = X[no_of_train_val:,:,:]
        y_train_val = y[:no_of_train_val,:]
        y_test = y[no_of_train_val:,:]
                    
        # Then create val subset from train set
        no_of_train = int((1-self.train_val_split) *\
                          X_train_val.shape[0])
                    
        X_train = X_train_val[:no_of_train,:,:]
        X_val = X_train_val[no_of_train:,:,:]
        y_train = y_train_val[:no_of_train,:]
        y_val = y_train_val[no_of_train:,:]
        
        print("Data was loaded!")
        print('Total no. of samples: ' + str(self.X.shape[0]))
        print('No. of training samples: ' + str(X_train.shape[0]))
        print('No. of validation samples: ' + str(X_val.shape[0]))
        print('No. of test samples: ' + str(X_test.shape[0]))
        print('Shape of each sample : '
              + str(X_train.shape[1]) + ' features (X)' 
              + ' + ' + str(y_train.shape[1])
              + ' labels (y)')
        
        if 'aug_values' in kwargs.keys():
            # Also split the arrays in the 'aug_values' dictionary.
            aug_values = kwargs['aug_values']
            shift_x = aug_values['shift_x']
            noise = aug_values['noise']
            fwhm = aug_values['fwhm']
            
            shift_x_train_val = shift_x[:no_of_train_val,:]
            shift_x_test = shift_x[no_of_train_val:,:]
            noise_train_val = noise[:no_of_train_val,:]
            noise_test = noise[no_of_train_val:,:]
            fwhm_train_val = fwhm[:no_of_train_val,:]
            fwhm_test = fwhm[no_of_train_val:,:]
            
            shift_x_train = shift_x_train_val[:no_of_train,:]
            shift_x_val = shift_x_train_val[no_of_train:,:]
            noise_train = noise_train_val[:no_of_train,:]
            noise_val = noise_train_val[no_of_train:,:]
            fwhm_train = fwhm_train_val[:no_of_train,:]
            fwhm_val = fwhm_train_val[no_of_train:,:]
            
            aug_values_train = {'shift_x' : shift_x_train,
                                'noise' : noise_train,
                                'fwhm' : fwhm_train}
            aug_values_val = {'shift_x' : shift_x_val,
                              'noise' : noise_val,
                              'fwhm' : fwhm_val}
            aug_values_test = {'shift_x' : shift_x_test,
                              'noise' : noise_test,
                              'fwhm' : fwhm_test}
            
            if 'scatterer' in aug_values.keys():
                # Also split the scatterer, distance, and pressure
                # arrays if they are present in the in the 'aug_values'
                # dictionary.
                scatterer = aug_values['scatterer']
                distance = aug_values['distance']
                pressure = aug_values['pressure']
                
                scatterer_train_val = scatterer[:no_of_train_val,:]
                scatterer_test = scatterer[no_of_train_val:,:]
                distance_train_val = distance[:no_of_train_val,:]
                distance_test = distance[no_of_train_val:,:]
                pressure_train_val = pressure[:no_of_train_val,:]
                pressure_test = pressure[no_of_train_val:,:]
            
                scatterer_train = scatterer_train_val[:no_of_train,:]
                scatterer_val = scatterer_train_val[no_of_train:,:]
                distance_train = distance_train_val[:no_of_train,:]
                distance_val = distance_train_val[no_of_train:,:]
                pressure_train = pressure_train_val[:no_of_train,:]
                pressure_val = pressure_train_val[no_of_train:,:]
                
                aug_values_train['scatterer'] = scatterer_train
                aug_values_train['distance'] = distance_train
                aug_values_train['pressure'] = pressure_train
                
                aug_values_val['scatterer'] = scatterer_val
                aug_values_val['distance'] = distance_val
                aug_values_val['pressure'] = pressure_val
                
                aug_values_test['scatterer'] = scatterer_test
                aug_values_test['distance'] = distance_test
                aug_values_test['pressure'] = pressure_test
                
            return X_train, X_val, X_test, y_train, y_val, y_test,\
                   aug_values_train, aug_values_val, \
                   aug_values_test
        
        elif 'names' in kwargs.keys():
            # Also split the names array.
            names = kwargs['names']
            names_train_val = names[:no_of_train_val,:]
            names_test = names[no_of_train_val:,:]
            
            names_train = names_train_val[:no_of_train,:]
            names_val = names_train_val[no_of_train:,:]
            
            return X_train, X_val, X_test, y_train, y_val, y_test,\
                   names_train, names_val, names_test
                   
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
        
    def check_class_distribution(self, task):
        data_list = [self.y,
                     self.y_train,
                     self.y_val,
                     self.y_test]
        self.class_distribution = ClassDistribution(task,
                                                    data_list)
        return self.class_distribution.cd
        
    def plot_random(self, 
                    no_of_spectra, 
                    dataset,
                    with_prediction,
                    loss_func=None): 
        """
        Plots random XPS spectra out of one of the data set.
        The labels and additional information are shown as texts on the
        plots.

        Parameters
        ----------
        no_of_spectra : int
            No. of plots to create.
        dataset : str, optional
            Either 'train', 'val', or 'test'.
            The default is 'train'.
        with_prediction : bool, optional
            If True, information about the predicted values are also 
            shown in the plot. 
            The default is False.

        Returns
        -------
        None.

        """    
        data = [] 
        texts = []
        
        if dataset == 'train':
            X = self.X_train
            y = self.y_train
            if with_prediction:
                pred = self.pred_train

        elif dataset == 'val':
            X = self.X_val
            y = self.y_val

        elif dataset == 'test':
            X = self.X_test
            y = self.y_test
            if with_prediction:
                pred = self.pred_test
                               
        for i in range(no_of_spectra):                
            r = np.random.randint(0, X.shape[0])
            if self.intensity_only:
                new_energies = np.reshape(np.array(self.energies),(-1,1))
                data.append(np.hstack((new_energies, X[r])))
            else:
                data.append(X[r])
            label = str(np.around(y[r], decimals = 3))
          
            text = ('Real: ' + label + '\n')
                
            if with_prediction:
                # Round prediction and sum to 1
                tmp_array = np.around(pred[r], decimals = 4)
                row_sums = tmp_array.sum()
                tmp_array = tmp_array / row_sums
                tmp_array = np.around(tmp_array, decimals = 3)    
                pred_text = ('Prediction: ' +\
                             str(list(tmp_array)) + '\n')
                try:
                    pred_label = ('Predicted label: ' +\
                                  str(self.pred_test_classes[r,0]))
                    pred_text += pred_label
                except AttributeError:
                    pass   
                text += pred_text 
                
                losses = [loss_func(y[i], pred[i]).numpy() \
                          for i in range(y.shape[0])]   
                loss_text = ('Loss: ' + str(np.around(losses[r],
                                                      decimals = 3)))
                text += loss_text                               
            try:
                aug_text = self._write_aug_text(
                    dataset = dataset,
                    index = r)
                text += aug_text
            except AttributeError:
                pass
            try:
                name_text = self._write_measured_text(
                    dataset = dataset,
                    index = r)
                text += name_text
            except AttributeError:
                pass
                
            if with_prediction:
                text += loss_text
                
            texts.append(text)
            
        data = np.array(data)
        
        graphic = SpectraPlot(data=data,
                              annots=texts)
        fig, axs = graphic.plot()    
        
    
    def show_worst_predictions(self, no_of_spectra, loss_func):
        losses = [loss_func(self.y_test[i], self.pred_test[i]).numpy() \
                  for i in range(self.y_test.shape[0])]
        
        worst_indices = [j[1] for j in 
                         sorted([(x,i) for (i,x) in enumerate(losses)],
                                reverse=True )[:no_of_spectra]]
        
        data = []
        texts = []
              
        for i in range(no_of_spectra):
            index = worst_indices[i] 
            if self.intensity_only:
                new_energies = np.reshape(np.array(self.energies),(-1,1))
                data.append(np.hstack((new_energies, self.X_test[index])))
            else:
                data.append(self.X_test[index])
                       
            label = str(np.around(self.y_test[index],
                                  decimals = 3))
            real = ('Real: ' +  label + '\n')
            text = real
            
            tmp_array = np.around(self.pred_test[index], 
                                  decimals = 3) 
            pred = ('Prediction: ' + str(list(tmp_array)) + '\n')
            text += pred
            
            try:
                aug = self._write_aug_text(
                    dataset = 'test',
                    index = index)
                text += aug
            except AttributeError:
                pass
            try:
                name = self._write_measured_text(
                    dataset = 'test',
                    index = index)
                text += name
            except AttributeError:
                pass
            loss_text = ('Loss: ' + str(np.around(losses[index],
                                                  decimals = 3)))
            text += loss_text
            
            texts.append(text)
            
        data = np.array(data)
        
        graphic = SpectraPlot(data=data,
                              annots=texts)
        fig, axs = graphic.plot()
        
    def show_wrong_classification(self):
        """
        Plots all spectra in the test data set for which a wrong class
        prediction was produced during training.

        Returns
        -------
        None.

        """
        data = []
        texts = []
        
        wrong_pred_args = []
        
        for i in range(self.pred_test.shape[0]): 
            argmax_class_true = np.argmax(self.y_test[i,:], axis = 0)
            argmax_class_pred = np.argmax(self.pred_test[i,:], axis = 0)
            
        if argmax_class_true != argmax_class_pred:
            wrong_pred_args.append(i)
        no_of_wrong_pred = len(wrong_pred_args)
        print('No. of wrong predictions on the test data: ' +\
              str(no_of_wrong_pred))
        
        if no_of_wrong_pred > 0:
            for i in range(no_of_wrong_pred):
                index = wrong_pred_args[i]
            
                real_y = ('Real: ' + \
                    str(self.y_test[index]) + '\n')
                # Round prediction and sum to 1
                tmp_array = np.around(self.pred_test[index], decimals = 4)
                row_sums = tmp_array.sum()
                tmp_array = tmp_array / row_sums
                tmp_array = np.around(tmp_array, decimals = 2)
                pred_y = ('Prediction: ' +\
                          str(tmp_array) + '\n')
                pred_label = ('Predicted label: ' +\
                              str(self.pred_test_classes[index,0]) + '\n')
                labels = self.y_test[index]
                for j, value in enumerate(labels):
                    if value == 1:
                        label = str(self.label_values[j])
                        label =  ('Real label: ' + label + '\n')
                text = real_y + pred_y + label + pred_label     
                try:
                    aug = self._write_aug_text(
                        dataset = 'test',
                        index = index)
                    text += aug
                except AttributeError:
                    pass
                try:
                    name = self._write_measured_text(
                        dataset = 'test',
                        index = index)
                    text += name
                except AttributeError:
                       pass
                
                texts.append(text)
            
            data = np.array(data)
        
            graphic = SpectraPlot(data=data,
                                  annots=texts)
            fig, axs = graphic.plot()
        
        
    def _write_aug_text(self, dataset, index):
        """
        Helper method for writing information about the parameters used 
        for data set creation into a figure. 

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct aug_values.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        aug_text : str
            Output text in a figure.

        """
        if dataset == 'train':
            shift_x = self.aug_values_train['shift_x'][index]
            noise = self.aug_values_train['noise'][index]
            fwhm = self.aug_values_train['fwhm'][index]
            
        elif dataset == 'val':
            shift_x = self.aug_values_val['shift_x'][index]
            noise = self.aug_values_val['noise'][index]
            fwhm = self.aug_values_val['fwhm'][index]
            
        elif dataset == 'test':
            shift_x = self.aug_values_test['shift_x'][index]
            noise = self.aug_values_test['noise'][index]
            fwhm = self.aug_values_test['fwhm'][index]
        
        if (fwhm is not None and fwhm != 0):
            fwhm_text = 'FHWM: ' + \
                    str(np.round(float(fwhm), decimals = 2)) + ', '
        else:
            fwhm_text = 'FHWM: not changed' + ', '
                
        if (shift_x is not None and shift_x != 0):            
            shift_text = ' Shift: ' + \
                    '{:.2f}'.format(float(shift_x)) + ', '
        else:
            shift_text = ' Shift: none' + ', '
                
        if (noise is not None and noise != 0):
            noise_text = 'S/N: ' + '{:.1f}'.format(float(noise))
        else:
            noise_text = 'S/N: not changed'
                
        aug_text = fwhm_text + shift_text + noise_text + '\n'
        
        if 'scatterer' in self.aug_values.keys():
            aug_text += self._write_scatter_text(dataset, index)
        else:
            aug_text += 'Spectrum not scattered.'
            
    
        return aug_text
    
    def _write_scatter_text(self, dataset, index):
        """
        Helper method for writing information about the parameters used 
        for simulation of scattering in a gas phase. 

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct aug_values.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        scatter_text : str
            Output text in a figure.

        """
        if dataset == 'train':
            scatterer = self.aug_values_train['scatterer'][index]
            distance = self.aug_values_train['distance'][index]
            pressure = self.aug_values_train['pressure'][index]
            
        elif dataset == 'val':
            scatterer = self.aug_values_val['scatterer'][index]
            distance = self.aug_values_val['distance'][index]
            pressure = self.aug_values_val['pressure'][index]
            
        elif dataset == 'test':
            scatterer = self.aug_values_test['scatterer'][index]
            distance = self.aug_values_test['distance'][index]
            pressure = self.aug_values_test['pressure'][index]
       
        scatterers = {'0' : 'He',
                      '1' : 'H2', 
                      '2' : 'N2',
                      '3' : 'O2'}       
        scatterer_name =  scatterers[str(scatterer[0])]

        name_text = 'Scatterer: ' + scatterer_name + ', '
        
        pressure_text = '{:.1f}'.format(float(pressure)) + ' mbar, '  
        
        distance_text = 'd = ' + \
                    '{:.1f}'.format(float(distance)) + ' mm'
                
        scatter_text = name_text + pressure_text + distance_text + '\n'
    
        return scatter_text

    def _write_measured_text(self, dataset, index):
        """
        Helper method for writing information about the measured 
        spectra in a data set into a figure. 

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct names.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        measured_text : str
            Output text in a figure.

        """
        if dataset == 'train':
            name = self.names_train[index][0]
            
        elif dataset == 'val':
            name = self.names_val[index][0]
            
        elif dataset == 'test':
            name = self.names_test[index][0]
        
        measured_text = 'Spectrum no. ' + str(index) + '\n' +\
                str(name) + '\n'
    
        return measured_text            
    
    
if __name__ == "__main__":
    np.random.seed(502)
    input_filepath = r'C:\Users\pielsticker\Simulations\20210222_Fe_linear_combination_small_gas_phase.h5'
    datahandler = DataHandler(intensity_only=True)
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 1000
    
    X_train, X_val, X_test, y_train, y_val, y_test,\
    aug_values_train, aug_values_val, aug_values_test =\
        datahandler.load_data_preprocess(
            input_filepath = input_filepath,
            no_of_examples = no_of_examples,
            train_test_split = train_test_split,
            train_val_split = train_val_split)        
    print('Input shape: ' + str(datahandler.input_shape))
    print('Labels: ' + str(datahandler.labels))
    print('No. of classes: ' + str(datahandler.num_classes))
  
    datahandler.plot_random(no_of_spectra = 20,
                            dataset = 'train',
                            with_prediction = False)