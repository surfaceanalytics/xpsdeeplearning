# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:32:48 2020

@author: Mark
"""

from .vamas_parser import VamasParser
from .text_parser import TextParser

class DataConverter():
    """ This class parses files of type: 
            'ProdigyXY', 'Vamas'
        And writes files of type:
            'JSON', 'Vamas', 'Excel'
    """
    
    def __init__(self):
        """ All objects of the Dataset class should have the methods specificed
        in the attribute 'self.class_methods'
        """
        self._parser_methods = {'Vamas': VamasParser,
                                'Text': TextParser}
        self._extensions = {'vms': 'Vamas',
                            'txt':'Text'}
        
    def load(self, filename, **kwargs):
        """ This method parses an input file an places it into a nested 
        dictionary.
        Parameters
        ----------
        filename: STRING
            The location and name of the file you wish to parse.
        **kwargs: 
            in_format: The file format of the loaded file.
        """
        if 'in_format' not in kwargs.keys():
            in_format = self._extensions[filename.rsplit('.',1)[-1].lower()]
        else:
            in_format = kwargs['in_format']
        
        self.parser = self._parser_methods[in_format]()
        self.data = self.parser.parse_file(filename)