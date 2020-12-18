# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:44 2020

@author: pielsticker
"""

import os
import shelve
import numpy as np
import json
import csv
import h5py
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping,\
    ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras import backend as K

try:
    import xpsdeeplearning.network.models as models
except:
    import network.models as models
    

#%%
class Classifier():
    """
    Class for training and testing a Keras model. Handles logging,
    saving, and loading automatically based on the time and the name
    of the loaded  data set.
    """
    def __init__(self, time, data_name = '', labels = []):
        """
        Initialize folder strucutre and an empty model.

        Parameters
        ----------
        time : str
            Time when the experiment is initiated.
        data_name : str, optional
            Name of the data set. Should include sufficient information
            for distinguishing runs on different data on the same day.
            The default is ''.
        labels : list
            List of strings of the labels of the XPS spectra.
            Example: 
                labels = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']

        Returns
        -------
        None.

        """
        self.time = time
        self.data_name = data_name
        self.label_values = labels
        
        self.model = Model()
        
        self.num_classes = len(self.label_values)
        
        root_dir = os.getcwd()
        dir_name = self.time + '_' + self.data_name 
        
        self.model_dir = os.path.join(*[root_dir, 'saved_models', dir_name])
        self.log_dir = os.path.join(*[root_dir, 'logs', dir_name])
        self.fig_dir = os.path.join(*[root_dir, 'figures', dir_name])
        
        if os.path.isdir(self.model_dir) == False:
            os.makedirs(self.model_dir)
            if os.path.isdir(self.model_dir) == True:
                print('Model folder created at ' +\
                      str(self.model_dir.split(root_dir)[1]))
        else:
            print('Model folder was already at ' +\
                  str(self.model_dir.split(root_dir)[1]))
        
        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir)
            if os.path.isdir(self.log_dir) == True:
                print('Logs folder created at ' +\
                      str(self.log_dir.split(root_dir)[1]))
        else:
            print('Logs folder was already at ' +\
                  str(self.log_dir.split(root_dir)[1]))
        
        if os.path.isdir(self.fig_dir) == False:
            os.makedirs(self.fig_dir)
            if os.path.isdir(self.fig_dir) == True:
                print('Figures folder created at ' +\
                      str(self.fig_dir.split(root_dir)[1]))
        else:
            print('Figures folder was already at ' +\
                  str(self.fig_dir.split(root_dir)[1]))
                
                
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
            dataset_size = hf['X'].shape[0]
            # Randomly choose a subset of the whole data set.
            r = np.random.randint(0, dataset_size-self.no_of_examples)
            X = hf['X'][r:r+self.no_of_examples, :, :]
            y = hf['y'][r:r+self.no_of_examples, :]
            
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
                self.input_shape = (self.X_train.shape[1], 1)
            
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
                self.input_shape = (self.X_train.shape[1], 1)
                                    
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
                self.input_shape = (self.X_train.shape[1], 1)
            
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
                pressure_val =pressure_train_val[no_of_train:,:]
                
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
    
    
    def summary(self):
        """
        Print a summary of the keras Model in self.model.
        Save the string value of the summary to the self.model_summary
        attribute.

        Returns
        -------
        None.

        """
        
        print(self.model.summary())
        
        #Save model summary to a string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)


    def save_and_print_model_image(self): 
        """
        Plot the model using the plot_model method from keras.
        Save the image to a file in the figure directory.

        Returns
        -------
        None.

        """
        fig_file_name = os.path.join(self.fig_dir, 'model.png')
        plot_model(self.model, to_file = fig_file_name,
                   rankdir = "LR", show_shapes = True,
                   show_layer_names = True)
        model_plot = plt.imread(fig_file_name)
        fig, ax = plt.subplots(figsize=(18, 2))
        ax.imshow(model_plot, interpolation='nearest')
        plt.tight_layout()
        plt.show()
        

    def train(self, checkpoint = True, early_stopping = False,
              tb_log = False, csv_log = True, epochs = 200, batch_size = 32,
              verbose = 1, new_learning_rate = None):
        """
        Train the keras model. Implements various callbacks during
        the training. Also checks the csv_log for previous training and
        appends accordingly.

        Parameters
        ----------
        checkpoint : bool, optional
            Determines if the model is saved when the val_loss is lowest.
            The default is True.
        early_stopping : bool, optional
            If early_stopping, the training is stopped when the val_loss
            increases for three straight epochs.
            The default is False.
        tb_log : bool, optional
            Controls the logging in Tensorboard.
            The default is False.
        csv_log : bool, optional
            If True, the data for each epoch is logged in a CSV file.
            Needs to be True if later retraining is planned.
            The default is True.
        epochs : int, optional
            Number of epochs to be trained. The default is 200.
        batch_size : int, optional
            Batch size for stochastic optimization.
            The default is 32.
        verbose : int, optional
            Controls the verbose parameter of the keras output.
            If 1, a progress log bar is shown in the output. 
            The default is 1.
        new_learning_rate : float, optional
            In case of retraining, the learning rate of the optimizer can
            be overwritten.
            The default is None.

        Returns
        -------
        self.history: dict
            Dictionary containing the training results for each epoch.

        """
        self.epochs = epochs
        self.batch_size = batch_size
        # In case of refitting, get the new epoch start point.
        epochs_trained = self._count_epochs_trained()
        
        if new_learning_rate != None:
            # Overwrite the previus optimizer learning rate using the
            # backend of keras.
            K.set_value(self.model.optimizer.learning_rate,
                        new_learning_rate)
            print('New learning rate: ' +\
                  str(K.eval(self.model.optimizer.lr)))

            
        callbacks = []
        
        if checkpoint:
            # Save the model if a new best validation_loss is achieved.
            model_file_name = self.model_dir
            checkpoint_callback = ModelCheckpoint(filepath = model_file_name,
                                                  monitor = 'val_loss',
                                                  verbose = 0,
                                                  save_best_only = True ,
                                                  save_weights_only = False,
                                                  mode = 'auto',
                                                  save_freq = 'epoch')
            callbacks.append(checkpoint_callback)

        if early_stopping:
            # Stop the training if the validation loss increases for 
            # three epochs.
            es_callback = EarlyStopping(monitor = 'val_loss',
                                        min_delta = 0, 
                                        patience = 3,
                                        verbose = True,
                                        mode = 'auto',
                                        baseline = None,
                                        restore_best_weights = True)
            callbacks.append(es_callback)
        
        if tb_log:
            # Log all training data in a format suitable
            # for TensorBoard anaylsis.
            tb_callback = TensorBoard(log_dir = self.log_dir,
                                      histogram_freq = 1,
                                      write_graph = True,
                                      write_images = False,
                                      profile_batch = 0)
            callbacks.append(tb_callback)
            
        if csv_log:   
            # Log the results of each epoch in a csv file.
            csv_file = os.path.join(self.log_dir,'log.csv')
            
            csv_callback = CSVLogger(filename = csv_file,
                                      separator=',',
                                      append = True)
            callbacks.append(csv_callback)
        
        # In case of submodel inputs, allow multiple inputs.
        X_train_data = []
        X_val_data = []
        for i in range(self.model.no_of_inputs):
            X_train_data.append(self.X_train)
            X_val_data.append(self.X_val)

        try:
            # Train the model and store the previous and the new
            # results in the history attribute.
            training = self.model.fit(X_train_data,
                                      self.y_train,
                                      validation_data = \
                                          (X_val_data, self.y_val),
                                          epochs = self.epochs +\
                                              epochs_trained,
                                      batch_size = self.batch_size,
                                      initial_epoch = epochs_trained,
                                      verbose = verbose,
                                      callbacks = callbacks) 
            
            self.last_training = training.history
            self.history = self._get_total_history()
            print('Training done!')
           
        except KeyboardInterrupt:
            # Save the model and the history in case of interruption.
            self.save_model()
            self.history = self._get_total_history()
            print('Training interrupted!')
        
        return self.history


    def evaluate(self):
        """
        Evaluate the Model on the test data.

        Returns
        -------
        score: dict or float
            If the accuracy is calculated, both the test loss and the
            test accuracy are returned. If not, only the test loss 
            is returned.
        """
        # In case of submodel inputs, allow multiple inputs.
        X_test_data = []
        
        for i in range(self.model.no_of_inputs):
            X_test_data.append(self.X_test)

        self.score = self.model.evaluate(X_test_data,
                                         self.y_test,
                                         batch_size = self.batch_size,
                                         verbose = True)      
        print('Evaluation done! \n')

        try:
            self.test_loss, self.test_accuracy = self.score[0], self.score[1]
        except:
            self.test_loss = self.score
            
        return self.score
     
     
    def predict(self, verbose = True):
        """
        Predict the labels on both the train and tests sets.

        Parameters
        ----------
        verbose : bool, optional
            If True, show the progress bar during prediction.
            The default is True.

        Returns
        -------
        pred_train : ndarray
            Array containing the predictions on the training set.
        pred_test : ndarray
            Array containing the predictions on the test set.

        """
        # In case of submodel inputs, allow multiple inputs.
        X_train_data = []
        X_test_data = []
        for i in range(self.model.no_of_inputs):
            X_train_data.append(self.X_train)
        
            X_test_data.append(self.X_test) 
        
        self.pred_train = self.model.predict(X_train_data, verbose = verbose)
        self.pred_test = self.model.predict(X_test_data, verbose = verbose)
        
        if verbose == True:
            print('Prediction done!')
        
        return self.pred_train, self.pred_test
    
    
    def _count_epochs_trained(self):
        """
        Count the numbers of previously trained epochs by searching the 
        csv log file

        Returns
        -------
        epochs : int
            No. of epochs for which the model was trained.

        """
        csv_file = os.path.join(self.log_dir,'log.csv')
        epochs = 0
        
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    epochs += 1                    
        except:
            pass
        
        return epochs
        
    def save_model(self,  **kwargs): 
        """
        Saves the model to the model directory. The model is saved both
        as a SavedModel object from Keras as well as a JSON file 
        containing the model parameters and the weights serialized to 
        HDF5.

        Returns
        -------
        None.

        """
        model_file_name = os.path.join(self.model_dir,
                                       'model.json')
        weights_file_name = os.path.join(self.model_dir,
                                         'weights.h5')
                        
        model_json = self.model.to_json()
        with open(model_file_name, 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_file_name)
        try:
            self.model.save(self.model_dir)
        except:
            pass
        print("Saved model to disk.")

    
    def save_hyperparams(self):
        """
        Save parameters of the training process to a JSON file. Used 
        for introspection and report writing. Only works after 
        training.

        Returns
        -------
        None.

        """
        hyperparam_file_name = os.path.join(self.model_dir,
                                            'hyperparameters.json')
        params_dict = {
            'model_name' : self.data_name,
            'datetime' : self.time,
            'Input file' : self.input_filepath,
            'model_summary' : self.model_summary,
            'num_of_classes' : self.num_classes,
            'train_test_split_percentage' : self.train_test_split,
            'train_val_split_percentage' : self.train_val_split,
            'epochs_trained' : self._count_epochs_trained(),
            'batch_size' : self.batch_size,
            'class_distribution' : self.check_class_distribution(),
            'loss' : type(self.model.loss).__name__,
            'optimizer' : self.model.optimizer._name,
            'learning rate' : str(K.eval(self.model.optimizer.lr)),
            'Labels': self.label_values,
            'Total no. of samples' : str(self.X.shape[0]),
            'No. of training samples' : str(self.X_train.shape[0]),
            'No. of validation samples' : str(self.X_val.shape[0]),
            'No. of test samples' : str(self.X_test.shape[0]),
            'Shape of each sample' : str(self.X_train.shape[1]) + \
                ' features (X)' + ' + ' + str(self.y_train.shape[1])
                + ' labels (y)'}
            
        with open(hyperparam_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(params_dict, json_file, ensure_ascii=False, indent=4)
        print("Saved hyperparameters to file.")
        
        
    def load_model(self, model_path = None, drop_last_layers = None):
        """
        Reload the model from file.

        Parameters
        ----------
        model_path : str, optional
            If model_path != None, then the model is loaded from a
            filepath.
            If None, the model is loaded from the self.model_dir.
            The default is None.
        drop_last_layers : int, optional
            No. of layers to be dropped during the loading. Helpful for
            transfer learning.
            The default is None.

        Returns
        -------
        None.

        """
        if model_path != None:
            file_name = model_path
        else:
            file_name = self.model_dir
        
        # Add the current model to the custom_objects dict. This is 
        # done to ensure the load_model method from Keras works 
        # properly.
        import inspect
        custom_objects = {}
        custom_objects[str(type(self.model).__name__)] =\
            self.model.__class__
        for name, obj in inspect.getmembers(models):
            if inspect.isclass(obj):
                if obj.__module__.startswith('xpsdeeplearning.network.models'):
                    custom_objects[obj.__module__ + '.' + obj.__name__] = obj

        # Load from file.    
        loaded_model = load_model(file_name, custom_objects = custom_objects)
        
        # Instantiate a new EmptyModel and implement it with the loaded
        # parameters. Needed for dropping the layers while keeping all
        # parameters.
        self.model = models.EmptyModel(
            inputs = loaded_model.input,
            outputs = loaded_model.layers[-1].output,
            inputshape = self.input_shape,
            num_classes = self.num_classes,
            no_of_inputs = loaded_model._serialized_attributes[
                'metadata']['config']['no_of_inputs'],
            name = 'Loaded_Model')

        print("Loaded model from disk.")
      
        if drop_last_layers != None:
            # Remove the last layers and stored the other layers in a
            # new EmptyModel object.
            no_of_drop_layers = drop_last_layers + 1
                
            new_model = models.EmptyModel(
                inputs = self.model.input,
                outputs = self.model.layers[-no_of_drop_layers].output,
                inputshape = self.input_shape,
                num_classes = self.num_classes,
                no_of_inputs = self.model.no_of_inputs,
                name = 'Changed_Model')
                
            self.model = new_model
            
            if no_of_drop_layers == 0:
                print('No layers were dropped.\n')
                
            if no_of_drop_layers == 1:
                print('The last layer was dropped.\n')
                
            else:
                print('The last {} layers were dropped.\n'.format(
                    str(drop_last_layers)))
                
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
        
        if (fwhm != None and fwhm != 0):
            fwhm_text = 'FHWM: ' + \
                    str(np.round(float(fwhm), decimals = 2)) + ', '
        else:
            fwhm_text = 'FHWM: not changed' + ', '
                
        if (shift_x != None and shift_x != 0):            
            shift_text = ' Shift: ' + \
                    '{:.2f}'.format(float(shift_x)) + ', '
        else:
            shift_text = ' Shift: none' + ', '
                
        if (noise != None and noise != 0):
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
        
        
class ClassifierSingle(Classifier):
    """
    Classifier subclass for one-class classification.
    Implements accuracy calculations and class predictions.
    """
    def __init__(self, time, data_name = '', labels = []):
        """
        Initialize a normal Classifier object to create folders.

        Parameters
        ----------
        time : str
            Time when the experiment is initiated.
        data_name : str, optional
            Name of the data set. Should include sufficient information
            for distinguishing runs on different data on the same day.
            The default is ''.
        labels : list, optional
            List of string of the labels of the XPS spectra.
            Very important for class predictions.
            Example: 
                labels = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']

        Returns
        -------
        None.

        """
        super(ClassifierSingle, self).__init__(time = time,
                                               data_name = data_name,
                                               labels = labels)
    
    
    def check_class_distribution(self):
        """
        Calculate how many examples of each class are in the different
        data sets.

        Returns
        -------
        class_distribution : dict
            Nested Dictionary of the format {'all data': dict,
                                      'training data': dict,
                                      'validation data': dict,
                                      'test data': dict}.
            Each of the sub-dicts contains the number of examples for 
            each label.
            
        """
        class_distribution = {'all data': {},
                              'training data': {},
                              'validation data': {},
                              'test data': {}}
        
        for i in range(self.num_classes):
            class_distribution['all data'][str(i)] = 0
            class_distribution['training data'][str(i)] = 0
            class_distribution['validation data'][str(i)] = 0
            class_distribution['test data'][str(i)] = 0
             
        data_list = [self.y, self.y_train, self.y_val,self.y_test]
       
        for i, dataset in enumerate(data_list):
            key = list(class_distribution.keys())[i]
            
            for j in range(dataset.shape[0]): 
                argmax_class = np.argmax(dataset[j,:], axis = 0)
                class_distribution[key][str(argmax_class)] +=1
                
        self.class_distribution = class_distribution
                
        return self.class_distribution
        
    
    def plot_class_distribution(self):  
        """
        Plot of the class_distribution in the different data sets.
        Has to be called after check_class_distribution.

        Returns
        -------
        None
        
        """             
        data = []
        for k, v in self.class_distribution.items():
            data_list = []
            for key, value in v.items():
                data_list.append(value)
            data.append(data_list)
        data = np.transpose(np.array(data))
            
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = np.arange(len(self.class_distribution.keys()))*1.5
 
        for i in range(data.shape[0]):
            ax.bar(x + i*0.25, data[i], align='edge', width = 0.2)
        plt.title('Class distribution')
        plt.legend(self.label_values)   
        plt.xticks(ticks=x+.5, labels=list(self.class_distribution.keys()))
        plt.show()
        
        
    def plot_random(self, no_of_spectra, dataset = 'train',
                    with_prediction = False): 
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
            If True, information about the predicted value is also 
            shown in the plot. 
            The default is False.

        Returns
        -------
        None.

        """
        no_of_cols = 5
        no_of_rows = int(no_of_spectra/no_of_cols)
        if (no_of_spectra % no_of_cols) != 0:
            no_of_rows += 1
            
        fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
        plt.subplots_adjust(left = 0.125, bottom = 0.5,
                            right=4.8, top = no_of_rows,
                            wspace = 0.2, hspace = 0.2)
    
        for i in range(no_of_spectra):
            x = np.arange(694, 750.05, 0.05)

            if dataset == 'train':
                r = np.random.randint(0, self.X_train.shape[0])
                y = self.X_train[r]
                labels = self.y_train[r]
                aug = self._write_aug_text(dataset = dataset, index = r)
                
                if with_prediction == True:
                    real_y = ('Real: ' + \
                            str(self.y_train[r]) + '\n')
                    # Round prediction and sum to 1
                    tmp_array = np.around(self.pred_train[r], decimals = 4)
                    row_sums = tmp_array.sum()
                    tmp_array = tmp_array / row_sums
                    tmp_array = np.around(tmp_array, decimals = 3)    
                    pred_y = ('Prediction: ' +\
                              str(tmp_array) + '\n')
                    pred_label = ('Predicted label: ' +\
                                  str(self.pred_train_classes[r,0]))

            elif dataset == 'val':
                r = np.random.randint(0, self.X_val.shape[0])
                y = self.X_val[r]
                labels = self.y_val[r]
                aug = self._write_aug_text(dataset = dataset, index = r)
                
            elif dataset == 'test':
                r = np.random.randint(0, self.X_test.shape[0])
                y = self.X_test[r]
                labels = self.y_test[r]
                aug = self._write_aug_text(dataset = dataset, index = r)
                
                if with_prediction == True:
                    real_y = ('Real: ' + \
                              str(self.y_test[r]) + '\n')
                    # Round prediction and sum to 1
                    tmp_array = np.around(self.pred_test[r], decimals = 4)
                    row_sums = tmp_array.sum()
                    tmp_array = tmp_array / row_sums
                    tmp_array = np.around(tmp_array, decimals = 3)    
                    pred_y = ('Prediction: ' +\
                              str(tmp_array) + '\n')
                    pred_label = ('Predicted label: ' +\
                                  str(self.pred_test_classes[r,0]))
                                  
            for j, value in enumerate(labels):
                if value == 1:
                    label = str(self.label_values[j] + '\n')
                    if with_prediction == True:
                        label =  ('Real label: ' + label)
                    
            row, col = int(i/no_of_cols), i % no_of_cols
            axs[row, col].plot(np.flip(x),y)
            axs[row, col].invert_xaxis()
            axs[row, col].set_xlim(750.05,694)
            axs[row, col].set_xlabel('Binding energy (eV)')
            axs[row, col].set_ylabel('Intensity (arb. units)')                          

            if with_prediction == False:
                axs[row, col].text(0.025, 0.3, label + aug,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12) 
            else:
                axs[row, col].text(0.025, 0.95,
                                   label + real_y + aug,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12) 
                axs[row, col].text(0.025, 0.3, pred_y + pred_label,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12) 
            
                
    def predict_classes(self):
        """
        Predicts the labels of all spectra in the training and test sets
        by checking which class has the maximum associated prediction
        value.

        Returns
        -------
        pred_train_classes : ndarray
            Array with the predicted classes on the training set.
        pred_test_classes : ndarray
            Array with the predicted classes on the test set.

        """
        pred_train, pred_test = self.predict(verbose = False)
        
        pred_train_classes = []
        pred_test_classes = []
        
        for i in range(pred_train.shape[0]): 
            argmax_class = np.argmax(pred_train[i,:], axis = 0)
            pred_train_classes.append(self.label_values[argmax_class])
            
        for i in range(pred_test.shape[0]): 
            argmax_class = np.argmax(pred_test[i,:], axis = 0)
            pred_test_classes.append(self.label_values[argmax_class])
            
        self.pred_train_classes = np.array(pred_train_classes).reshape(-1,1)
        self.pred_test_classes = np.array(pred_test_classes).reshape(-1,1)
        
        print('Class prediction done!')
        
        return self.pred_train_classes, self.pred_test_classes
    

    def show_wrong_classification(self):
        """
        Plots all spectra in the test data set for which a wrong class
        prediction was produced during training.

        Returns
        -------
        None.

        """
        binding_energy = np.arange(694, 750.05, 0.05)
        
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
            no_of_cols = 5
            no_of_rows = int(no_of_wrong_pred/no_of_cols)
            if (no_of_wrong_pred % no_of_cols) != 0:
                no_of_rows += 1

            fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
            plt.subplots_adjust(left = 0.125, bottom = 0.5,
                                right=4.8, top = no_of_rows,
                                wspace = 0.2, hspace = 0.2)

            for n in range(no_of_wrong_pred):
                arg = wrong_pred_args[n]
                intensity = self.X_test[arg]
            
                real_y = ('Real: ' + \
                    str(self.y_test[arg]) + '\n')
                aug = self._write_aug_text(dataset = 'test', index = arg)
                # Round prediction and sum to 1
                tmp_array = np.around(self.pred_test[arg], decimals = 4)
                row_sums = tmp_array.sum()
                tmp_array = tmp_array / row_sums
                tmp_array = np.around(tmp_array, decimals = 2)
                pred_y = ('Prediction: ' +\
                          str(tmp_array) + '\n')
                pred_label = ('Predicted label: ' +\
                              str(self.pred_test_classes[arg,0]) + '\n')
                labels = self.y_test[arg]
                for j, value in enumerate(labels):
                    if value == 1:
                        label = str(self.label_values[j])
                        label =  ('Real label: ' + label + '\n')
                
                text = real_y + pred_y + label + pred_label + aug
            
                row, col = int(n/no_of_cols), n % no_of_cols
                axs[row, col].plot(np.flip(binding_energy),intensity)
                axs[row, col].invert_xaxis()
                axs[row, col].set_xlim(750.05,694)
                axs[row, col].set_xlabel('Binding energy (eV)')
                axs[row, col].set_ylabel('Intensity (arb. units)')  
                axs[row, col].text(0.025, 0.4, text,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12)
                
    def _get_total_history(self):
        """
        Loads the previous training history from the CSV log file.
        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing all previous training results from 
            this classifier instance.

        """
        csv_file = os.path.join(self.log_dir,'log.csv')
        history = {'accuracy' : [],
                   'loss' : [],
                   'val_accuracy': [],
                   'val_loss': []}
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d = dict(row)
                    history['accuracy'].append(float(d['accuracy']))
                    history['loss'].append(float(d['loss']))
                    history['val_accuracy'].append(float(d['val_accuracy']))
                    history['val_loss'].append(float(d['val_loss']))                
        except:
            pass
        
        return history
            
            
    def shelve_results(self, full = False):
        """
        Store all inputs and outputs into a shelve object.
        
        Parameters
        ----------
        full : bool, optional
            If full, the spectral features, the class distribution and
            the training history are also stored. The default is False.

        Returns
        -------
        None.

        """
        filename = os.path.join(self.model_dir,'vars')
        
        with shelve.open(filename,'n') as shelf:
            key_list = ['y_train', 'y_test', 'pred_train', 'pred_test',
                    'pred_train_classes', 'pred_test_classes',
                    'test_accuracy', 'test_loss', 'aug_values']
            if full == True:
                key_list.extend(['X, X_train', 'X_val', 'X_test', 'y',
                                 'y_val', 'class_distribution', 'hist'])
            for key in key_list:
                shelf[key] = vars(self)[key]
        
        print("Saved results to file.")
            
            

class ClassifierMultiple(Classifier):
    """
    Classifier subclass for mult-class regression.
    """
    def __init__(self, time, data_name = '', labels = []):
        """
        Initialize a normal Classifier object to create folders.

        Parameters
        ----------
        time : str
            Time when the experiment is initiated.
        data_name : str, optional
            Name of the data set. Should include sufficient information
            for distinguishing runs on different data on the same day.
            The default is ''.
        labels : list, optional
            List of string of the labels of the XPS spectra.
            Very important for class predictions.
            Example: 
                labels = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']

        Returns
        -------
        None.

        """
        super(ClassifierMultiple, self).__init__(time = time,
                                                 data_name = data_name,
                                                 labels = labels)
        

    def check_class_distribution(self):
        """
        Calculate the average distibutions of the labels in the 
        different
        data sets.

        Returns
        -------
        class_distribution : dict
             Dictionary of the format {'all data': dict,
                                       'training data': dict,
                                       'validation data': dict,
                                       'test data': dict}.
            Each of the sub-dicts contains the average distribution of 
            the labels in the data sub-set.
            
        """
        class_distribution = {'all data': {},
                                'training data': {},
                                'validation data': {},
                                'test data': {}}
        
        for i in range(self.num_classes):
            class_distribution['all data'][str(i)] = 0
            class_distribution['training data'][str(i)] = 0
            class_distribution['validation data'][str(i)] = 0
            class_distribution['test data'][str(i)] = 0
             
        data_list = [self.y, self.y_train, self.y_val,self.y_test]
       
        for i, dataset in enumerate(data_list):
            key = list(class_distribution.keys())[i]
            average = list(np.mean(dataset, axis = 0))
            class_distribution[key] = average
                
        self.class_distribution = class_distribution
                
        return self.class_distribution

    
    def plot_class_distribution(self):      
        """
        Plot of the average label distribution in the different data sets.
        Has to be called after check_class_distribution.

        Returns
        -------
        None
        
        """             
        data = []
        for k, v in self.class_distribution.items():
            data.append(v)
        data = np.transpose(np.array(data))     
            
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = np.arange(len(self.class_distribution.keys()))*1.5
 
        for i in range(data.shape[0]):
            ax.bar(x + i*0.25, data[i], align='edge', width = 0.2)
        plt.title('Average distribution')
        plt.legend(self.label_values)   
        plt.xticks(ticks=x+.5, labels=list(self.class_distribution.keys()))
        plt.show()
        
        
    def plot_random(self, no_of_spectra, dataset = 'train',
                    with_prediction = False): 
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
        no_of_cols = 5
        no_of_rows = int(no_of_spectra/no_of_cols)
        if (no_of_spectra % no_of_cols) != 0:
            no_of_rows += 1
            
        fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
        plt.subplots_adjust(left = 0.125, bottom = 0.5,
                            right=4.8, top = no_of_rows,
                            wspace = 0.2, hspace = 0.2)
    
        for i in range(no_of_spectra):
            energies = np.arange(694, 750.05, 0.05)

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
                
            r = np.random.randint(0, X.shape[0])
            intensity = X[r]
            label = str(np.around(y[r], decimals = 3))
 
            full_text = ('Real: ' + label + '\n')
            full_text_pred = full_text
                
            if with_prediction:
                # Round prediction and sum to 1
                tmp_array = np.around(pred[r], decimals = 4)
                row_sums = tmp_array.sum()
                tmp_array = tmp_array / row_sums
                tmp_array = np.around(tmp_array, decimals = 3)    
                pred_text = ('Prediction: ' +\
                             str(list(tmp_array)) + '\n')
                full_text_pred = full_text + pred_text
                loss = self.model.loss
                losses = [loss(y[i], pred[i]).numpy() \
                          for i in range(y.shape[0])]
                               
                loss_text = ('Loss: ' + str(np.around(losses[r],
                                                      decimals = 3)))
                                               
            try:
                aug_text = self._write_aug_text(dataset = dataset, index = r)
                full_text += aug_text
                full_text_pred += aug_text
            except AttributeError:
                pass
            try:
                name_text = self._write_measured_text(dataset = dataset,
                                                      index = r)
                full_text += name_text
                full_text_pred += name_text
            except AttributeError:
                pass
                
            if with_prediction:
                full_text_pred += loss_text
                            
            row, col = int(i/no_of_cols), i % no_of_cols
            axs[row, col].plot(np.flip(energies),intensity)
            axs[row, col].invert_xaxis()
            axs[row, col].set_xlim(750.05,694)
            axs[row, col].set_xlabel('Binding energy (eV)')
            axs[row, col].set_ylabel('Intensity (arb. units)')         

            if with_prediction == False:
                axs[row, col].text(0.025, 0.4, full_text,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12) 
            else:
                axs[row, col].text(0.025, 0.4, full_text_pred,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12)

    
    def _get_total_history(self):
        """
        Loads the previous training history from the CSV log file.
        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing all previous training results from 
            this classifier instance.

        """
        csv_file = os.path.join(self.log_dir,'log.csv')
        history = {'loss' : [],
                   'val_loss': []}
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d = dict(row)
                    history['loss'].append(float(d['loss']))
                    history['val_loss'].append(float(d['val_loss']))
        except:
            pass
        
        return history
        
                
    def show_worst_predictions(self, no_of_spectra):
        """
        Plots the highest associated loss in the test data set.

        Parameters
        ----------
        no_of_spectra : int
            No. of spectra to plot.

        Returns
        -------
        None.

        """        
        loss = self.model.loss
        losses = [loss(self.y_test[i], self.pred_test[i]).numpy() \
                  for i in range(self.y_test.shape[0])]
        
        worst_indices = [j[1] for j in 
                         sorted([(x,i) for (i,x) in enumerate(losses)],
                                reverse=True )[:no_of_spectra]]

        no_of_cols = 5
        no_of_rows = int(no_of_spectra/no_of_cols)
        if (no_of_spectra % no_of_cols) != 0:
            no_of_rows += 1

        fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
        plt.subplots_adjust(left = 0.125, bottom = 0.5,
                            right=4.8, top = no_of_rows,
                            wspace = 0.2, hspace = 0.2)                    
    
        for i in range(no_of_spectra):
            index = worst_indices[i]
            x = np.arange(694, 750.05, 0.05)
            y = self.X_test[index]
            label = str(np.around(self.y_test[index], decimals = 3))
            real = ('Real: ' +  label + '\n')
            full_text = real
            
            tmp_array = np.around(self.pred_test[index], decimals = 3) 
            pred = ('Prediction: ' + str(list(tmp_array)) + '\n')
            full_text += pred
            
            try:
                aug = self._write_aug_text(dataset = 'test', index = index)
                full_text += aug
            except AttributeError:
                pass
            try:
                name = self._write_measured_text(dataset = 'test', index = index)
                full_text += name
            except AttributeError:
                pass
            loss_text = ('Loss: ' + str(np.around(losses[index], decimals = 3)))
            full_text += loss_text

            row, col = int(i/no_of_cols), i % no_of_cols
            axs[row, col].plot(np.flip(x),y)
            axs[row, col].invert_xaxis()
            axs[row, col].set_xlim(750.05,694)
            axs[row, col].set_xlabel('Binding energy (eV)')
            axs[row, col].set_ylabel('Intensity (arb. units)')                          
            axs[row, col].text(0.025, 0.4, full_text,
                               horizontalalignment='left',
                               verticalalignment='top',
                               transform = axs[row, col].transAxes,
                               fontsize = 12)


    def shelve_results(self, full = False):
        """
        Store all inputs and outputs into a shelve object.
        
        Parameters
        ----------
        full : bool, optional
            If full, the spectral features, the class distribution and
            the training history are also stored. The default is False.

        Returns
        -------
        None.

        """
        filename = os.path.join(self.model_dir,'vars')
        
        with shelve.open(filename,'n') as shelf:
            key_list = ['y_train', 'y_test', 'pred_train', 'pred_test',
                        'test_loss']
            try:
                key_list.append('aug_values_train')
                key_list.append('aug_values_test')
            except:
                pass
            try:
                key_list.append('names_train')
                key_list.append('names_test')

            except:
                pass
              
            if full == True:
                key_list.extend(['X, X_train', 'X_val', 'X_test', 'y',
                                 'y_val', 'class_distribution', 'history'])
            for key in key_list:
                shelf[key] = vars(self)[key]
        
        print("Saved results to file.")