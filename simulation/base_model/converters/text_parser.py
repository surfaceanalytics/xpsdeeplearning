# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:15:57 2020

@author: Mark
"""
import numpy as np

#%%
class TextParser():
    def __init__(self, **kwargs):
        if 'n_headerlines' in kwargs.keys():
            self.n_headerlines = kwargs['n_headerlines']
        else:
            self.n_headerlines = 8
        self.data_dict = []
            
    def parse_file(self, filepath):
        """ The openFile method parses the .xy file into a list of dictionaries 
        under the attribute 'self.data'.
        Each dictionary is a grouping of related attributes.
        These are later put into a heirarchical nested dictionary that 
        represents the native data structure of the export, and is well 
        represented by JSON.
        """
        self._read_lines(filepath)
        self._parse_header()
        return self._build_dict()
    
    def _read_lines(self, filepath):
        self.data = []
        self.filepath = filepath
        with open(filepath) as fp:
            for line in fp:
               self.data += [line]
   
    def _parse_header(self):
        self.header = self.data[:self.n_headerlines]
        self.data = self.data[self.n_headerlines:]
        
    def _build_dict(self):
        lines = np.array([[float(i) for i in d.split()] for d in self.data])
        x = lines[:,0]
        y = lines[:,1]
        x,y = self._check_step_width(x,y)
        spect = {'data':{'x':list(x), 'y0':list(y)}}
        self.data_dict += [spect]
        return self.data_dict
            
    def _check_step_width(self, x, y):
        start = x[0]
        stop = x[-1]
        x1 = np.roll(x,-1)
        diff = np.abs(np.subtract(x,x1))
        step = round(np.min(diff[diff!=0]),2)
        if (stop-start)/step > len(x):
            x, y = self._interpolate(x, y, step)
        return x, y
    
    def _interpolate(self, x, y, step):
        new_x = []
        new_y = []
        for i in range(len(x)-1):
            diff = np.abs(np.around(x[i+1]-x[i],2))
            if (diff > step) & (diff < 10):
                for j in range(int(np.round(diff/step))):
                    new_x += [x[i] + j*step]
                    k = j / int(diff/step)
                    new_y += [y[i]*(1-k) + y[i+1]*k]
            else:
                new_x += [x[i]]
                new_y += [y[i]]
                
        new_x += [x[-1]]
        new_y += [y[-1]]
        x = new_x
        y = np.array(new_y)
        return x, y