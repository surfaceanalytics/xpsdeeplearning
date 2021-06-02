# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:32:48 2020.

@author: Mark
"""

from .vamas_parser import VamasParser
from .text_parser import TextParser


class DataConverter:
    """Class for loading/writing XPS data of different formats."""

    def __init__(self):
        """
        Initialize the parser methods.
        
        All objects of the Dataset class should have the methods specificed
        in the attribute 'self.class_methods'
        
        This class parses files of type: 
            'ProdigyXY', 'Vamas'
        And writes files of type:
            'JSON', 'Vamas', 'Excel'
        """
        self._parser_methods = {"Vamas": VamasParser, "Text": TextParser}
        self._extensions = {"vms": "Vamas", "txt": "Text"}

    def load(self, filename, **kwargs):
        """
        Parse an input file.
        
        The result is placed into a nested dictionary.
        
        Parameters
        ----------
        filename: STRING
            The location and name of the file you wish to parse.
        **kwargs: 
            in_format: The file format of the loaded file.
        """
        if "in_format" not in kwargs.keys():
            in_format = self._extensions[filename.rsplit(".", 1)[-1].lower()]
        else:
            in_format = kwargs["in_format"]

        self.parser = self._parser_methods[in_format]()
        self.data = self.parser.parse_file(filename)
