# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:41:38 2020

@author: pielsticker
"""

# imports
# =============================================================================
# import os
# import sys
# import numpy as np
# import pandas as pd
# from keras.utils import np_utils
# from keras.optimizers import Adam
# from keras.models import Sequential
# from matplotlib import pyplot as plt
# from keras.callbacks import ModelCheckpoint
# from sklearn.cross_validation import train_test_split
# from keras.layers.pooling import GlobalAveragePooling1D
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Dropout, Activation, Dense, Flatten
# from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
# from keras.utils import plot_model
# =============================================================================

# best model
# =============================================================================
# np.random.seed(seed)
# best_model_file = "weights/CNN_Noise_DataAug/highest_val_acc_weights_epoch{epoch:02d}-val_acc{val_acc:.3f}_"+str(argv[1])+".h5"
# best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
# training = model.fit(X_train,
#                      y_train,
#                      validation_data = (X_test, y_test),
#                      nb_epoch = self.epochs,
#                      batch_size = self.batch_size,
#                      callbacks = [best_model],
#                      shuffle = True,
#                      verbose=1)
# =============================================================================
        
        
        
# preprocessing
# =============================================================================
#     def load_data_preprocess():
#         #load data
#         path_to_input = 'input_spectra'
#         Mn2_C = pd.read_pickle(os.path.join(path_to_input, 'Mn2_Larger_Clean_Thin.pkl'))
#         Mn3_C = pd.read_pickle(os.path.join(path_to_input, 'Mn3_Larger_Clean_Thin.pkl'))
#         Mn4_C = pd.read_pickle(os.path.join(path_to_input, 'Mn4_Larger_Clean_Thin.pkl'))
#         Mn_All = (Mn2_C.append(Mn3_C, ignore_index=True)).append(Mn4_C, ignore_index=True)
#         Mn_All = np.array(Mn_All)
# 
#         labels = make_labels(Mn2_C, Mn3_C, Mn4_C)
# 
#         X_train, X_test, y_train, y_test = train_test_split(Mn_All, labels, test_size=self.train_test_percent, random_state=13)
#         
# 
# 
#         if crop_spectra == True:
#             X_train, X_test = crop(X_train, X_test, min = 100, max = 600)
# 
#         X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, mean_center = True, norm = True )
#     
#         return X_train, X_test, y_train, y_test
# 
#     def make_labels(Mn2_C, Mn3_C, Mn4_C):
#         labels=[]
#         for i in range(len(Mn2_C)):
#             labels.append(0)
#         for i in range(len(Mn3_C)):
#             labels.append(1)
#         for i in range(len(Mn4_C)):
#             labels.append(2)
#         
#         return labels
# 
#     def crop(X_train, X_test, min = 100,max = 600):
#         crop_X_train = X_train[:,min:max]
#         crop_X_test = X_test[:,min:max]
#         
#         return crop_X_train, crop_X_test
# 
#     def preprocess(X_train, X_test, y_train, y_test, mean_center = False, norm = True):
#         X_train = np.array(X_train).astype('float32')
#         X_train = X_train.reshape(X_train.shape + (1,))
#         X_test = np.array(X_test).astype('float32')
#         X_test = X_test.reshape(X_test.shape + (1,))
#     
#         if mean_center == True:
#             X_train -=  np.mean(X_train)
#             X_test -= np.mean(X_test)
#             print( 'Data mean-centered')
#         if norm == True:
#             X_train /= np.max(X_train)
#             X_test /= np.max(X_test)
#             print( 'Data normalized')
# 
#         y_train = np.array(y_train)
#         y_test = np.array(y_test)
#         y_train = np_utils.to_categorical(y_train)
#         y_test = np_utils.to_categorical(y_test)
#         print( 'Data one-hot encoded')
# 
#         print("Total of "+str(y_test.shape[1])+" classes.")
#         print("Total of "+str(len(X_train))+" training samples.")
#         print("Total of "+str(len(X_test))+" testing samples.")
#     
#         return X_train, X_test, y_train, y_test
# =============================================================================