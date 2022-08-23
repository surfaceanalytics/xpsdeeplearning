# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:32:48 2020.

@author: Mark
"""

from .vamas_parser import VamasParser
from .text_parser import TextParser
from .writers import TextWriter, VamasWriter

#%%


class DataConverter:
    """Class for loading/writing XPS data of different formats."""

    def __init__(self):
        """
        Initialize the parser methods.

        All objects of the Dataset class should have the methods
        specificed in the attribute 'self.class_methods'.

        This class parses files of type:
            'Text', 'Vamas'
        And writes files of type:
            'Text', 'Vamas', ''
        """
        self._parsers = {"Vamas": VamasParser, "Text": TextParser}
        self._writers = {"Text": TextWriter, "Vamas": VamasWriter}
        self._extensions = {"vms": "Vamas", "txt": "Text"}

    def load(self, filename, **kwargs):
        """
        Parse an input file.

        The result is placed into a nested dictionary.

        Parameters
        ----------
        filename: str
            The location and name of the file you wish to parse.
        **kwargs:
            in_format: The file format of the loaded file.
        """
        if "in_format" not in kwargs.keys():
            in_format = self._extensions[filename.rsplit(".", 1)[-1].lower()]
        else:
            in_format = kwargs["in_format"]

        self.parser = self._parsers[in_format]()
        self.data = self.parser.parse_file(filename)

    def write(self, filename, **kwargs):
        """
        Write data to new file.

        Parameters
        ----------
        filename: str
            The name of the new file..
        **kwargs:
            out_format: The file format of the output file.

        Returns
        -------
        None.

        """
        if "out_format" not in kwargs.keys():
            out_format = self._extensions[filename.rsplit(".", 1)[-1].lower()]
        else:
            out_format = kwargs["out_format"]

        self.writer = self._writers[out_format]()
        data = self.data
        self.writer.write(data, filename)
