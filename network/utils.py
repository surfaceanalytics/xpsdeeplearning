# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:44 2020

@author: pielsticker
"""
import os
import shelve
import numpy as np
from matplotlib import pyplot as plt

class TrainingGraphs():
    def __init__(self, history, model_name, time):
        self.history = history
        root_dir = os.getcwd().partition('network')[0]
        
        self.fig_file_name = root_dir + '\\figures\\' +\
            time + '_' + model_name + '\\'
        
        self.plot_loss()
        self.plot_accuracy()
        
    def plot_loss(self):
        # summarize history for loss
        plt.figure(figsize=(9, 5))
        plt.plot(self.history['loss'], linewidth = 3)
        plt.plot(self.history['val_loss'], linewidth = 3)
        plt.title('Loss')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        fig_name = self.fig_file_name + 'loss.png' 
        plt.savefig(fig_name)
        plt.show()

    def plot_accuracy(self):
        #summarize history for accuracy
        plt.figure(figsize=(9, 5))
        plt.plot(self.history['accuracy'], linewidth = 3)
        plt.plot(self.history['val_accuracy'], linewidth = 3)
        plt.title('Accuracy')
        plt.ylabel('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        fig_name = self.fig_file_name + 'accuracy.png' 
        plt.savefig(fig_name)
        plt.show()
        
        
        
class SpectrumFigure:
    def __init__(self, y, fig_text):
        self.x = np.arange(694,750.05,0.05)
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(5,4), dpi=100)
        self.fig.patch.set_facecolor('0.9411')
        self.ax.plot(np.flip(self.x),self.y)
        self.ax.invert_xaxis()
        self.ax.set_xlabel('Binding energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')
        self.fig.tight_layout()
        self.fig_text = fig_text
        self.ax.text(0.1, 0.45,fig_text,
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform = self.ax.transAxes,
                     fontsize = 9)



def shelve_all(time, model_name):
    root_dir = os.getcwd().partition('network')[0]
    dir_name = time + '_' + model_name 
    filename = root_dir + '\\saved_models\\' + dir_name + '\\vars'
    
    with shelve.open(filename,'n') as shelf:
        for key in ['X', 'X_train', 'X_val', 'X_test', 'y', 'y_train',
                    'y_val', 'y_test', 'pred_train', 'pred_test',
                    'pred_train_classes', 'pred_test_classes',
                    'test_accuracy', 'test_loss', 'class_distribution',
                    'hist', 'pred_test']:
            try:
                shelf[key] = globals()[key]
            except:
                # __builtins__, shelf, and imported modules can not
                # be shelved.
                print('ERROR shelving: {0}'.format(key))
            
        return shelf