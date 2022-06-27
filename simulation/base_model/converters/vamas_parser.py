# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:29:54 2020.

@author: Mark
"""
import numpy as np

from .vamas import VamasHeader, Block

#%%
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
        Parse .xy into a list of dictionaries caleld 'self.data'.

        Each dictionary is a grouping of related attributes.
        These are later put into a heirarchical nested dictionary that
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

        with open(filepath, "rb") as fp:
            for line in fp:
                if line.endswith(b"\r\n"):
                    self.data += [line.decode("utf-8").strip()]

    def _parse_header(self):
        """
        Parse the vamas header into a VamasHeader object.

        The common_header_attr are the header attributes that are common
        to both types of Vama format (NORM and MAP).

        Returns
        -------
        None.

        """
        for attr in self.common_header_attr:
            setattr(self.header, attr, self.data.pop(0).strip())
        n = int(self.header.noCommentLines)
        comments = ""
        for line_no in range(n):
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
        for v in range(int(self.header.nrExpVar)):
            for attr in self.exp_var_attributes:
                setattr(self.header, attr, self.data.pop(0).strip())

    def _parse_blocks(self):
        """
        Parse all blocks in the vamas data.

        Returns
        -------
        None.

        """
        for b in range(int(self.header.noBlocks)):
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
        # print('Block instantiated in time: ' + str(stop-start))

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
        for n in range(block.noCommentLines):
            block.commentLines += self.data.pop(0)
        block.technique = self.data.pop(0).strip()
        for v in range(int(self.header.nrExpVar)):
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

        # stop = time.time()
        # print('Block metadata added in time: ' + str(stop-start))

        # start = time.time()
        self._add_data_values(block)
        # stop = time.time()
        # print('Block data added in time: ' + str(stop-start))

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
        for n in range(block.noCommentLines):
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

        for v in range(block.noVariables):
            name = "y" + str(v)
            data_dict[name] = []

        d = list(
            np.array(self.data[: block.numOrdValues], dtype=np.float32)
        )

        self.data = self.data[block.numOrdValues :]

        # =============================================================================
        #         for r in range(int(block.numOrdValues / block.noVariables)):
        #             for v in range(block.noVariables):
        #                 name = 'y' + str(v)
        #                 data_dict[name] += [float(self.data.pop(0).strip())]
        # =============================================================================

        for v in range(block.noVariables):
            n = block.noVariables
            name = "y" + str(v)
            dd = d[v::n]
            data_dict[name] = dd
            setattr(block, name, data_dict[name])

    def _build_dict(self):
        """
        Construct a list of dictionaries.

        Each dictionary contains all the data and metadata of a spectrum.
        vamas.sampleID -> group['name']
        vamas.
        """
        group_id = -1
        temp_group_name = ""
        spectra = []

        for idx, b in enumerate(self.blocks):
            group_name = b.sampleID

            # This set of conditions detects if the group name has
            # changed. If it has, then it increments the group_idx.
            if group_name != temp_group_name:
                temp_group_name = group_name
                group_id += 1

            spectrum_type = str(b.speciesLabel + b.transitionLabel)
            spectrum_id = idx

            settings = {
                "analysis_method": b.technique,
                "dwell_time": b.dwellTime,
                "workfunction": b.workFunction,
                "excitation_energy": b.sourceEnergy,
                "pass_energy": b.resolution,
                "scan_mode": b.analyzerMode,
                "source_label": b.sourceLabel,
                "nr_values": int(b.numOrdValues / b.noVariables),
                "x_units": b.abscissaLabel,
                "y_units": b.variableLabel1,
            }

            date = (
                str(b.year)
                + "-"
                + str(b.month)
                + "-"
                + str(b.day)
                + " "
                + str(b.hour)
                + ":"
                + str(b.minute)
                + ":"
                + str(b.second)
            )

            data = {"x": b.x}
            for n in range(int(b.noVariables)):
                key = "y" + str(n)
                data[key] = getattr(b, key)

            spec_dict = {
                "date": date,
                "group_name": group_name,
                "group_id": group_id,
                "spectrum_type": spectrum_type,
                "spectrum_id": spectrum_id,
                "scans": b.noScans,
                "settings": settings,
                "data": data,
            }

            spectra += [spec_dict]

        self.data_dict = spectra
        return self.data_dict


#%%
if __name__ == "__main__":
    filepath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning\data\references\NiCoFe\Fe2p_Fe_metal.vms"
    v = VamasParser()
    v.parse_file(filepath)
    h = v.header
    n = h.noBlocks
    header = h.__dict__
