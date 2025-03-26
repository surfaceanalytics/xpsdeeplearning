#
# Copyright the xpsdeeplearning authors.
#
# This file is part of xpsdeeplearning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Dataconverter for XPS data.
"""
from xpsdeeplearning.simulation.base_model.converters.text_parser import TextParser
from xpsdeeplearning.simulation.base_model.converters.vamas_parser import VamasParser
from xpsdeeplearning.simulation.base_model.converters.writers import (
    TextWriter,
    VamasWriter,
)


class DataConverter:
    """Class for loading/writing XPS data of different formats."""

    def __init__(self):
        """
        Initialize the parser methods.

        All objects of the Dataset class should have the methods
        specificed in the attribute "self.class_methods".

        This class parses files of type:
            "Text", "Vamas"
        And writes files of type:
            "Text", "Vamas", ""
        """
        self._parsers = {"Vamas": VamasParser, "Text": TextParser}
        self._writers = {"Text": TextWriter, "Vamas": VamasWriter}
        self._extensions = {"vms": "Vamas", "txt": "Text"}

        self.parser = None
        self.data = None
        self.writer = None

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
