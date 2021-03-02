# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:44 2020

@author: pielsticker
"""

import os
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from .data_handling import DataHandler
from .exp_logging import ExperimentLogging

from . import models
    
#%%
class Classifier():
    """
    Class for training and testing a Keras model. Handles logging,
    saving, and loading automatically based on the time and the experiment
    name.
    """
    def __init__(self,
                 time,
                 exp_name = '',
                 task = 'regression',
                 intensity_only = True,
                 labels = []):
        """
        Initialize folder structure and an empty model.

        Parameters
        ----------
        time : str
            Time when the experiment is initiated.
        exp_name : str, optional
            Name of the experiment. Should include sufficient information
            for distinguishing runs on different data on the same day.
            The default is ''.
        labels : list
            List of strings of the labels of the XPS spectra.
            Can also be loaded from file.
            Example: 
                labels = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']

        Returns
        -------
        None.

        """
        self.time = time
        self.exp_name = exp_name
        self.task = task
        self.intensity_only = intensity_only
        
        self.model = Model()
          
        self.datahandler = DataHandler(intensity_only)
        self.datahandler.labels = labels     
        self.datahandler.num_classes = len(self.datahandler.labels)
        
        dir_name = self.time + '_' + self.exp_name 
        self.logging = ExperimentLogging(dir_name)
        self.logging.make_dirs()
        
        self.logging.hyperparams = {
            'time' : self.time,
            'exp_name' : self.exp_name,
            'task' : self.task,
            'intensity_only' : self.intensity_only
            }
        self.logging.save_hyperparams()
    
                
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
        fig_file_name = os.path.join(self.logging.fig_dir, 'model.png')
        plot_model(self.model, to_file = fig_file_name,
                   rankdir = "LR", show_shapes = True,
                   show_layer_names = True)
        model_plot = plt.imread(fig_file_name)
        fig, ax = plt.subplots(figsize=(18, 2))
        ax.imshow(model_plot, interpolation='nearest')
        plt.tight_layout()
        plt.show()
        
    def train(self,
              epochs,
              batch_size,
              checkpoint=True, 
              early_stopping=False,
              tb_log=False,
              csv_log=True,
              hyperparam_log=True,
              verbose=1,
              new_learning_rate = None):
        """
        Train the keras model. Implements various callbacks during
        the training. Also checks the csv_log for previous training and
        appends accordingly.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to be trained. The default is 200.
        batch_size : int, optional
            Batch size for stochastic optimization.
            The default is 32.
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
        hyperparam_log : bool, optional
            If True, the hyperparameters of the classifier are saved
            in a CSV file.
            Needs to be True if later retraining is planned.
            The default is True.
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
        self.batch_size = batch_size
        # In case of refitting, get the new epoch start point.
        epochs_trained = self.logging._count_epochs_trained()
        
        if new_learning_rate != None:
            # Overwrite the previus optimizer learning rate using the
            # backend of keras.
            K.set_value(self.model.optimizer.learning_rate,
                        new_learning_rate)
            print('New learning rate: ' +\
                  str(K.eval(self.model.optimizer.lr)))

        callback_list = self.logging.activate_cbs(
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            tb_log=tb_log,
            csv_log=csv_log,
            hyperparam_log=hyperparam_log)
        

            
        # Update jaosn file with hyperparameters.
        train_params = {
            'model_summary' : self.model_summary,
            'epochs_trained' : epochs_trained,
            'batch_size' : self.batch_size,
            'loss' : type(self.model.loss).__name__,
            'optimizer' : type(self.model.optimizer).__name__,
            'learning_rate' : str(K.eval(self.model.optimizer.lr))
            }  
        self.logging.update_saved_hyperparams(train_params)
                
        # In case of submodel inputs, allow multiple inputs.
        X_train_data = []
        X_val_data = []
        for i in range(self.model.no_of_inputs):
            X_train_data.append(self.datahandler.X_train)
            X_val_data.append(self.datahandler.X_val)

        try:
            # Train the model and store the previous and the new
            # results in the history attribute.
            training = self.model.fit(X_train_data,
                                      self.datahandler.y_train,
                                      validation_data = \
                                          (X_val_data,
                                           self.datahandler.y_val),
                                          epochs = epochs +\
                                              epochs_trained,
                                      batch_size = self.batch_size,
                                      initial_epoch = epochs_trained,
                                      verbose = verbose,
                                      callbacks = callback_list) 
            
            self.last_training = training.history
            self.history = self.logging._get_total_history(self.task)
            print('Training done!')
           
        except KeyboardInterrupt:
            # Save the model and the history in case of interruption.
            print('Training interrupted!')
            self.save_model()
            self.history = self.logging._get_total_history(self.task)
        
        epoch_param = {
            'epochs_trained' : self.logging._count_epochs_trained()
            }  
        self.logging.update_saved_hyperparams(epoch_param)
        
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
            X_test_data.append(self.datahandler.X_test)

        score = self.model.evaluate(X_test_data,
                                    self.datahandler.y_test,
                                    batch_size = self.batch_size,
                                    verbose = True)      
        print('Evaluation done! \n')

        try:
            self.datahandler.test_loss, self.datahandler.test_accuracy = \
                score[0], score[1]
        except:
            self.datahandler.test_loss = score
            
        return score
     
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
            X_train_data.append(self.datahandler.X_train)
            X_test_data.append(self.datahandler.X_test) 
        
        self.datahandler.pred_train = self.model.predict(X_train_data,
                                                         verbose = verbose)
        self.datahandler.pred_test = self.model.predict(X_test_data,
                                                        verbose = verbose)
        
        if verbose == True:
            print('Prediction done!')
        
        return self.datahandler.pred_train, self.datahandler.pred_test
    
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
        
        if self.task == 'regression':
            print('Regression was chosen as task. ' +\
                  'No prediction of classes possible!')
        else:
            pred_train_classes = []
            pred_test_classes = []
            
            for i in range(self.datahandler.pred_train.shape[0]): 
                argmax_class = np.argmax(self.datahandler.pred_train[i,:],
                                         axis = 0)
                pred_train_classes.append(
                    self.datahandler.label_values[argmax_class])
            
            for i in range(self.datahandler.pred_test.shape[0]): 
                argmax_class = np.argmax(self.datahandler.pred_test[i,:],
                                         axis = 0)
                pred_test_classes.append(
                    self.datahandler.label_values[argmax_class])
            
            self.datahandler.pred_train_classes = \
                np.array(pred_train_classes).reshape(-1,1)
            self.datahandler.pred_test_classes = \
                np.array(pred_test_classes).reshape(-1,1)
        
            print('Class prediction done!')
        
            return self.datahandler.pred_train_classes, \
                self.datahandler.pred_test_classes
            
    def save_model(self): 
        """
        Saves the model to the model directory. The model is saved both
        as a SavedModel object from Keras as well as a JSON file 
        containing the model parameters and the weights serialized to 
        HDF5.

        Returns
        -------
        None.

        """
        model_file_name = os.path.join(self.logging.model_dir,
                                       'model.json')
        weights_file_name = os.path.join(self.logging.model_dir,
                                         'weights.h5')
                        
        model_json = self.model.to_json()
        with open(model_file_name, 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_file_name)
        try:
            self.model.save(self.logging.model_dir)
        except:
            pass
        print("Saved model to disk.")
                
    def load_model(self, model_path = None, drop_last_layers = None):
        """
        Reload the model from file.

        Parameters
        ----------
        model_path : str, optional
            If model_path != None, then the model is loaded from a
            filepath.
            If None, the model is loaded from self.logging.model_dir.
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
            file_name = self.logging.model_dir
        
        # Add the current model to the custom_objects dict. This is 
        # done to ensure the load_model method from Keras works 
        # properly.
        import inspect
        custom_objects = {}
        custom_objects[str(type(self.model).__name__)] =\
            self.model.__class__
        for name, obj in inspect.getmembers(models):
            if inspect.isclass(obj):
                if obj.__module__.startswith(
                        'xpsdeeplearning.network.models'):
                    custom_objects[obj.__module__ + '.' + obj.__name__] = obj

        # Load from file.    
        loaded_model = load_model(file_name, custom_objects = custom_objects)
        
        # Instantiate a new EmptyModel and implement it with the loaded
        # parameters. Needed for dropping the layers while keeping all
        # parameters.
        self.model = models.EmptyModel(
            inputs = loaded_model.input,
            outputs = loaded_model.layers[-1].output,
            inputshape = self.datahandler.input_shape,
            num_classes = self.datahandler.num_classes,
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
                inputshape = self.datahandler.input_shape,
                num_classes = self.datahandler.num_classes,
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
                        
    def load_data_preprocess(self,
                             input_filepath,
                             no_of_examples,
                             train_test_split,
                             train_val_split):
        loaded_data = self.datahandler.load_data_preprocess(
            input_filepath,
            no_of_examples,
            train_test_split,
            train_val_split)
        
        data_params = {
            'input_filepath' : self.datahandler.input_filepath,
            'train_test_split' : self.datahandler.train_test_split,
            'train_val_split' : self.datahandler.train_val_split,
            'labels': self.datahandler.labels,
            'num_of_classes' : self.datahandler.num_classes,
            'no_of_examples' : self.datahandler.no_of_examples,
            'No. of training samples' : str(self.datahandler.X_train.shape[0]),
            'No. of validation samples' : str(self.datahandler.X_val.shape[0]),
            'No. of test samples' : str(self.datahandler.X_test.shape[0]),
            'Shape of each sample' : str(self.datahandler.X_train.shape[1]) + \
                ' features (X)' + ' + ' + str(self.datahandler.y_train.shape[1])
                + ' labels (y)'}
            
        self.logging.update_saved_hyperparams(data_params)
        
        return loaded_data
    
    def plot_random(self,
                    no_of_spectra,
                    dataset = 'train',
                    with_prediction = False): 
        if with_prediction:
            self.datahandler.plot_random(no_of_spectra,
                                         dataset,
                                         with_prediction = True,
                                         loss_func=self.model.loss)  
        else:
            self.datahandler.plot_random(no_of_spectra,
                                         dataset,
                                         with_prediction = False)     
    
    def show_worst_predictions(self, no_of_spectra):
        self.datahandler.show_worst_predictions(no_of_spectra,
                                                loss_func = self.model.loss)
        
    def show_wrong_classification(self):
        self.datahandler.show_wrong_classification()
                   
    def plot_class_distribution(self):
        self.datahandler.class_distribution.plot(self.datahandler.labels)
                   
    def pickle_results(self):
        """
        Pickles and saves all outputs.        

        Returns
        -------
        None.

        """
        filename = os.path.join(self.logging.log_dir,'results.pkl')
        
        key_list = ['y_train',
                    'y_test',
                    'pred_train',
                    'pred_test',
                    'test_loss',
                    'class_distribution']
        if self.task == 'classification':
            key_list.extend(['test_accuracy',
                             'pred_train_classes',
                             'pred_test_classes'])
        
        pickle_data = {}
        for key in key_list:
            try:
                if key == 'class_distribution':
                    cd = getattr(self.datahandler, key)
                    pickle_data[key] = cd.cd
                else:
                    pickle_data[key] = getattr(self.datahandler, key)
            except:
                pass            
        
        with open(filename,'n') as pickle_file:
            pickle.dump(pickle_data,
                        pickle_file,
                        protocol=pickle.HIGHEST_PROTOCOL)
            
        print("Saved results to file.")
        

        
def restore_clf_from_logs(logpath):
    hyperparam_file_name = os.path.join(logpath,
                                        'hyperparameters.json')
            
    with open(hyperparam_file_name, 'r') as json_file:
        hyperparams = json.load(json_file)
        exp_name = hyperparams['exp_name']
        time = hyperparams['time']
        task = hyperparams['task']
        intensity_only = hyperparams['intensity_only']
        
    clf = Classifier(time=time,
                     exp_name=exp_name,
                     task=task,
                     intensity_only=intensity_only)
    clf.logging.hyperparams = hyperparams
    clf.logging.save_hyperparams()
    
    print("Recovered classifier from file.")  
    
    return clf 

#%% 
if __name__ == "__main__":
    np.random.seed(502)
    time = '20210218_09h21m'
    exp_name = 'Fe_4_classes_linear_comb_new_noise_small_resnet'

    clf = Classifier(time = time,
                     exp_name = exp_name)
    input_filepath = r'C:\Users\pielsticker\Simulations\20210222_Fe_linear_combination_small_gas_phase.h5'
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 1000
    
    X_train, X_val, X_test, y_train, y_val, y_test,\
    aug_values_train, aug_values_val, aug_values_test =\
        clf.datahandler.load_data_preprocess(
            input_filepath = input_filepath,
            no_of_examples = no_of_examples,
            train_test_split = train_test_split,
            train_val_split = train_val_split)