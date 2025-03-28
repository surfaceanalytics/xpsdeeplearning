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
Dataconverter for XPS data in vms format.
"""

import numpy as np

from xpsdeeplearning.simulation.base_model.converters.vamas import Block, VamasHeader


class VamasParser:
    """Parser for XPS data stored in Vamas files."""

    def __init__(self):
        """
        Initialize VamasHeader and blocks list.

        Returns
        -------
        None.

        """
        self.header = VamasHeader()
        self.blocks = []
        self.common_header_attr = [
            "formatID",
            "instituteID",
            "instrumentModelID",
            "operatorID",
            "experimentID",
            "noCommentLines",
        ]

        self.exp_var_attributes = ["expVarLabel", "expVarUnit"]

        self.norm_header_attr = [
            "scanMode",
            "nrRegions",
            "nrExpVar",
            "unknown3",
            "unknown4",
            "unknown5",
            "unknown6",
            "noBlocks",
        ]

        self.map_header_attr = [
            "scanMode",
            "nrRegions",
            "nr_positions",
            "nr_x_coords",
            "nr_y_coords",
            "nrExpVar",
            "unknown3",
            "unknown4",
            "unknown5",
            "unknown6",
            "noBlocks",
        ]

        self.norm_block_attr = [
            "blockID",
            "sampleID",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "noHrsInAdvanceOfGMT",
            "noCommentLines",
            "commentLines",
            "technique",
            "expVarValue",
            "sourceLabel",
            "sourceEnergy",
            "unknown1",
            "unknown2",
            "unknown3",
            "sourceAnalyzerAngle",
            "unknown4",
            "analyzerMode",
            "resolution",
            "magnification",
            "workFunction",
            "targetBias",
            "analyzerWidthX",
            "analyzerWidthY",
            "analyzerTakeOffPolarAngle",
            "analyzerAzimuth",
            "speciesLabel",
            "transitionLabel",
            "particleCharge",
            "abscissaLabel",
            "abscissaUnits",
            "abscissaStart",
            "abscissaStep",
            "noVariables",
            "variableLabel1",
            "variableUnits1",
            "variableLabel2",
            "variableUnits2",
            "signalMode",
            "dwellTime",
            "noScans",
            "timeCorrection",
            "sampleAngleTilt",
            "sampleTiltAzimuth",
            "sampleRotation",
            "noAdditionalParams",
            "paramLabel1",
            "paramUnit1",
            "paramValue1",
            "paramLabel2",
            "paramUnit2",
            "paramValue2",
            "numOrdValues",
            "minOrdValue1",
            "maxOrdValue1",
            "minOrdValue2",
            "maxOrdValue2",
            "dataString",
        ]

        self.map_block_attr = [
            "blockID",
            "sampleID",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "noHrsInAdvanceOfGMT",
            "noCommentLines",
            "commentLines",
            "technique",
            "x_coord",
            "y_coord",
            "expVarValue",
            "sourceLabel",
            "sourceEnergy",
            "unknown1",
            "unknown2",
            "unknown3",
            "fov_x",
            "fovy",
            "sourceAnalyzerAngle",
            "unknown4",
            "analyzerMode",
            "resolution",
            "magnification",
            "workFunction",
            "targetBias",
            "analyzerWidthX",
            "analyzerWidthY",
            "analyzerTakeOffPolarAngle",
            "analyzerAzimuth",
            "speciesLabel",
            "transitionLabel",
            "particleCharge",
            "abscissaLabel",
            "abscissaUnits",
            "abscissaStart",
            "abscissaStep",
            "noVariables",
            "variableLabel1",
            "variableUnits1",
            "variableLabel2",
            "variableUnits2",
            "signalMode",
            "dwellTime",
            "noScans",
            "timeCorrection",
            "sampleAngleTilt",
            "sampleTiltAzimuth",
            "sampleRotation",
            "noAdditionalParams",
            "paramLabel1",
            "paramUnit1",
            "paramValue1",
            "paramLabel2",
            "paramUnit2",
            "paramValue2",
            "numOrdValues",
            "minOrdValue1",
            "maxOrdValue1",
            "minOrdValue2",
            "maxOrdValue2",
            "dataString",
        ]

    def parse_file(self, filepath):
        """
        Parse .xy into a list of dictionaries caleld "self.data".

        Each dictionary is a grouping of related attributes.
        These are later put into a hierarchical nested dictionary that
        represents the native data structure of the export, and is
        well represented by JSON.
        """
        self._read_lines(filepath)
        self._parse_header()
        self._parse_blocks()
        return self._build_dict()

    def _read_lines(self, filepath):
        """
        Read all lines in the Vamas file.

        Parameters
        ----------
        filepath : str
            Has to be a .vms file.

        Returns
        -------
        None.

        """
        self.data = []
        self.filepath = filepath

        with open(filepath, "rb") as vms_file:
            for line in vms_file:
                if line.endswith(b"\r\n") or line.endswith(b"\n"):
                    self.data += [line.decode("utf-8").strip()]

    def _parse_header(self):
        """
        Parse the vamas header into a VamasHeader object.

        The common_header_attr are the header attributes that are common
        to both types of Vamas format (NORM and MAP).

        Returns
        -------
        None.

        """
        for attr in self.common_header_attr:
            setattr(self.header, attr, self.data.pop(0).strip())
        n = int(self.header.noCommentLines)
        comments = ""
        for _ in range(n):
            comments += self.data.pop(0)
        self.header.commentLines = comments
        self.header.expMode = self.data.pop(0).strip()
        if self.header.expMode == "NORM":
            for attr in self.norm_header_attr:
                setattr(self.header, attr, self.data.pop(0).strip())
                if attr == "nrExpVar":
                    self._add_exp_var()

        elif self.header.expMode == "MAP":
            for attr in self.map_header_attr:
                setattr(self.header, attr, self.data.pop(0).strip())
                if attr == "nrExpVar":
                    self._add_exp_var()

    def _add_exp_var(self):
        """
        Add the attribute exp_var to the VamasHeader.

        Returns
        -------
        None.

        """
        for _ in range(int(self.header.nrExpVar)):
            for attr in self.exp_var_attributes:
                setattr(self.header, attr, self.data.pop(0).strip())

    def _parse_blocks(self):
        """
        Parse all blocks in the vamas data.

        Returns
        -------
        None.

        """
        for _ in range(int(self.header.noBlocks)):
            self._parseOneBlock()

    def _parseOneBlock(self):
        """
        Parse one block of vamas data.

        Depending on the experimental mode, a differnt method is used.

        Returns
        -------
        None.

        """
        if self.header.expMode == "NORM":
            self.blocks += [self._parse_NORM_Block()]
        elif self.header.expMode == "MAP":
            self.blocks += [self._parse_MAP_block()]

    def _parse_NORM_Block(self):
        """
        Parse a NORM block from Vamas.

        Returns
        -------
        block : vamas.BLOCK
            A Block object containing all data from one VAMAS block.

        """
        # start = time.time()
        block = Block()
        # stop = time.time()
        # print("Block instantiated in time: " + str(stop-start))

        # start = time.time()

        block.blockID = self.data.pop(0).strip()
        block.sampleID = self.data.pop(0).strip()
        block.year = int(self.data.pop(0).strip())
        block.month = int(self.data.pop(0).strip())
        block.day = int(self.data.pop(0).strip())
        block.hour = int(self.data.pop(0).strip())
        block.minute = int(self.data.pop(0).strip())
        block.second = int(self.data.pop(0).strip())
        block.noHrsInAdvanceOfGMT = int(self.data.pop(0).strip())
        block.noCommentLines = int(self.data.pop(0).strip())
        for _ in range(block.noCommentLines):
            block.commentLines += self.data.pop(0)
        block.technique = self.data.pop(0).strip()
        for _ in range(int(self.header.nrExpVar)):
            block.expVarValue = self.data.pop(0).strip()
        block.sourceLabel = self.data.pop(0).strip()
        block.sourceEnergy = float(self.data.pop(0).strip())
        block.unknown1 = self.data.pop(0).strip()
        block.unknown2 = self.data.pop(0).strip()
        block.unknown3 = self.data.pop(0).strip()
        block.sourceAnalyzerAngle = self.data.pop(0).strip()
        block.unknown4 = self.data.pop(0).strip()
        block.analyzerMode = self.data.pop(0).strip()
        block.resolution = float(self.data.pop(0).strip())
        block.magnification = self.data.pop(0).strip()
        block.workFunction = float(self.data.pop(0).strip())
        block.targetBias = float(self.data.pop(0).strip())
        block.analyzerWidthX = self.data.pop(0).strip()
        block.analyzerWidthY = self.data.pop(0).strip()
        block.analyzerTakeOffPolarAngle = self.data.pop(0).strip()
        block.analyzerAzimuth = self.data.pop(0).strip()
        block.speciesLabel = self.data.pop(0).strip()
        block.transitionLabel = self.data.pop(0).strip()
        block.particleCharge = self.data.pop(0).strip()
        block.abscissaLabel = self.data.pop(0).strip()
        block.abscissaUnits = self.data.pop(0).strip()
        block.abscissaStart = float(self.data.pop(0).strip())
        block.abscissaStep = float(self.data.pop(0).strip())
        block.noVariables = int(self.data.pop(0).strip())
        for param in range(block.noVariables):
            name = "variableLabel" + str(param + 1)
            setattr(block, name, self.data.pop(0).strip())
            name = "variableUnits" + str(param + 1)
            setattr(block, name, self.data.pop(0).strip())
        block.signalMode = self.data.pop(0).strip()
        block.dwellTime = float(self.data.pop(0).strip())
        block.noScans = int(self.data.pop(0).strip())
        block.timeCorrection = self.data.pop(0).strip()
        block.sampleAngleTilt = float(self.data.pop(0).strip())
        block.sampleTiltAzimuth = float(self.data.pop(0).strip())
        block.sampleRotation = float(self.data.pop(0).strip())
        block.noAdditionalParams = int(self.data.pop(0).strip())
        for param in range(block.noAdditionalParams):
            name = "paramLabel" + str(param + 1)
            setattr(block, name, self.data.pop(0))
            name = "paramUnit" + str(param + 1)
            setattr(block, name, self.data.pop(0))
            name = "paramValue" + str(param + 1)
            setattr(block, name, self.data.pop(0))
        block.numOrdValues = int(self.data.pop(0).strip())
        for param in range(block.noVariables):
            name = "minOrdValue" + str(param + 1)
            setattr(block, name, float(self.data.pop(0).strip()))
            name = "maxOrdValue" + str(param + 1)
            setattr(block, name, float(self.data.pop(0).strip()))

        # stop = time.time()
        # print("Block metadata added in time: " + str(stop-start))

        # start = time.time()
        self._add_data_values(block)
        # stop = time.time()
        # print("Block data added in time: " + str(stop-start))

        return block

    def _parse_MAP_block(self):
        """
        Parse a MAP block from Vamas.

        Returns
        -------
        block : vamas.BLOCK
            A Block object containing all data from one VAMAS block.

        """
        block = Block()
        block.blockID = self.data.pop(0).strip()
        block.sampleID = self.data.pop(0).strip()
        block.year = int(self.data.pop(0).strip())
        block.month = int(self.data.pop(0).strip())
        block.day = int(self.data.pop(0).strip())
        block.hour = int(self.data.pop(0).strip())
        block.minute = int(self.data.pop(0).strip())
        block.second = int(self.data.pop(0).strip())
        block.noHrsInAdvanceOfGMT = int(self.data.pop(0).strip())
        block.noCommentLines = int(self.data.pop(0).strip())
        for _ in range(block.noCommentLines):
            self.data.pop(0)
            block.commentLines += self.data.pop(0)
        block.technique = self.data.pop(0).strip()
        block.x_coord = self.data.pop(0).strip()
        block.y_coord = self.data.pop(0).strip()
        block.expVarValue = self.data.pop(0).strip()
        block.sourceLabel = self.data.pop(0).strip()
        block.sourceEnergy = float(self.data.pop(0).strip())
        block.unknown1 = self.data.pop(0).strip()
        block.unknown2 = self.data.pop(0).strip()
        block.unknown3 = self.data.pop(0).strip()
        block.fov_x = self.data.pop(0).strip()
        block.fov_y = self.data.pop(0).strip()
        block.sourceAnalyzerAngle = self.data.pop(0).strip()
        block.unknown4 = self.data.pop(0).strip()
        block.analyzerMode = self.data.pop(0).strip()
        block.resolution = float(self.data.pop(0).strip())
        block.magnification = self.data.pop(0).strip()
        block.workFunction = float(self.data.pop(0).strip())
        block.targetBias = float(self.data.pop(0).strip())
        block.analyzerWidthX = self.data.pop(0).strip()
        block.analyzerWidthY = self.data.pop(0).strip()
        block.analyzerTakeOffPolarAngle = self.data.pop(0).strip()
        block.analyzerAzimuth = self.data.pop(0).strip()
        block.speciesLabel = self.data.pop(0).strip()
        block.transitionLabel = self.data.pop(0).strip()
        block.particleCharge = self.data.pop(0).strip()
        block.abscissaLabel = self.data.pop(0).strip()
        block.abscissaUnits = self.data.pop(0).strip()
        block.abscissaStart = float(self.data.pop(0).strip())
        block.abscissaStep = float(self.data.pop(0).strip())
        block.noVariables = int(self.data.pop(0).strip())
        for p in range(block.noVariables):
            name = "variableLabel" + str(p + 1)
            setattr(block, name, self.data.pop(0).strip())
            name = "variableUnits" + str(p + 1)
            setattr(block, name, self.data.pop(0).strip())
        block.signalMode = self.data.pop(0).strip()
        block.dwellTime = float(self.data.pop(0).strip())
        block.noScans = int(self.data.pop(0).strip())
        block.timeCorrection = self.data.pop(0).strip()
        block.sampleAngleTilt = float(self.data.pop(0).strip())
        block.sampleTiltAzimuth = float(self.data.pop(0).strip())
        block.sampleRotation = float(self.data.pop(0).strip())
        block.noAdditionalParams = int(self.data.pop(0).strip())
        for p in range(block.noAdditionalParams):
            name = "paramLabel" + str(p + 1)
            setattr(block, name, self.data.pop(0))
            name = "paramUnit" + str(p + 1)
            setattr(block, name, self.data.pop(0))
            name = "paramValue" + str(p + 1)
            setattr(block, name, self.data.pop(0))
        block.numOrdValues = int(self.data.pop(0).strip())
        for p in range(block.noVariables):
            name = "minOrdValue" + str(p + 1)
            setattr(block, name, float(self.data.pop(0).strip()))
            name = "maxOrdValue" + str(p + 1)
            setattr(block, name, float(self.data.pop(0).strip()))

        self._add_data_values(block)

        return block

    def _add_data_values(self, block):
        """
        Add the data from one block to a dictionary.

        Parameters
        ----------
        block : vamas.Block
            Block object with the dictionary as an attribute.

        Returns
        -------
        None.

        """
        data_dict = {}
        start = float(block.abscissaStart)
        step = float(block.abscissaStep)
        num = int(block.numOrdValues / block.noVariables)
        x = [round(start + i * step, 2) for i in range(num)]

        if block.abscissaLabel == "binding energy":
            x.reverse()

        setattr(block, "x", x)

        for var in range(block.noVariables):
            name = "y" + str(var)
            data_dict[name] = []

        data_list = list(np.array(self.data[: block.numOrdValues], dtype=np.float32))

        self.data = self.data[block.numOrdValues :]

        for var in range(block.noVariables):
            max_var = block.noVariables
            name = "y" + str(var)
            data_one = data_list[var::max_var]
            data_dict[name] = data_one
            setattr(block, name, data_dict[name])

    def _build_dict(self):
        """
        Construct a list of dictionaries.

        Each dictionary contains all the data and metadata of a spectrum.
        vamas.sampleID -> group["name"]
        vamas.
        """
        group_id = -1
        temp_group_name = ""
        spectra = []

        for idx, block in enumerate(self.blocks):
            group_name = block.sampleID

            # This set of conditions detects if the group name has
            # changed. If it has, then it increments the group_idx.
            if group_name != temp_group_name:
                temp_group_name = group_name
                group_id += 1

            spectrum_type = str(block.speciesLabel + block.transitionLabel)
            spectrum_id = idx

            settings = {
                "analysis_method": block.technique,
                "dwell_time": block.dwellTime,
                "workfunction": block.workFunction,
                "excitation_energy": block.sourceEnergy,
                "pass_energy": block.resolution,
                "scan_mode": block.analyzerMode,
                "source_label": block.sourceLabel,
                "nr_values": int(block.numOrdValues / block.noVariables),
                "x_units": block.abscissaLabel,
                "y_units": block.variableLabel1,
            }

            date = (
                str(block.year)
                + "-"
                + str(block.month)
                + "-"
                + str(block.day)
                + " "
                + str(block.hour)
                + ":"
                + str(block.minute)
                + ":"
                + str(block.second)
            )

            data = {"x": block.x}
            for n in range(int(block.noVariables)):
                key = "y" + str(n)
                data[key] = getattr(block, key)

            spec_dict = {
                "date": date,
                "group_name": group_name,
                "group_id": group_id,
                "spectrum_type": spectrum_type,
                "spectrum_id": spectrum_id,
                "scans": block.noScans,
                "settings": settings,
                "data": data,
            }

            spectra += [spec_dict]

        self.data_dict = spectra
        return self.data_dict
