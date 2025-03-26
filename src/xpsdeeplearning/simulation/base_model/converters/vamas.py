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
Data model for XPS data in Vamas format.
"""


class VamasHeader:
    """Class for handling headers in Vamas files."""

    def __init__(self):
        """
        Define default values for some parameters in the header.

        Returns
        -------
        None.

        """
        self.formatID = (
            "VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4"
        )
        self.instituteID = "Not Specified"
        self.instriumentModelID = "Not Specified"
        self.operatorID = "Not Specified"
        self.experimentID = "Not Specified"
        self.noCommentLines = "2"
        self.commentLines = "Casa Info Follows CasaXPS Version 2.3.22PR1.0\n0"
        self.expMode = "NORM"
        self.scanMode = "REGULAR"
        self.nrRegions = "0"
        self.nrExpVar = "1"
        self.expVarLabel = "Exp Variable"
        self.expVarUnit = "d"
        self.unknown3 = "0"
        self.unknown4 = "0"
        self.unknown5 = "0"
        self.unknown6 = "0"
        self.noBlocks = "1"


class Block:
    """Class for handling blocks in Vamas files."""

    def __init__(self):
        """
        Define default values for some parameters of a block.

        Returns
        -------
        None.

        """
        self.blockID = ""
        self.sampleID = ""
        self.year = ""
        self.month = ""
        self.day = ""
        self.hour = ""
        self.minute = ""
        self.second = ""
        self.noHrsInAdvanceOfGMT = "0"
        self.noCommentLines = ""
        # This list should contain one element per
        # for each line in the comment block.
        self.commentLines = ""
        self.technique = ""
        self.expVarValue = ""
        self.sourceLabel = ""
        self.sourceEnergy = ""
        self.unknown1 = "0"
        self.unknown2 = "0"
        self.unknown3 = "0"
        self.sourceAnalyzerAngle = ""
        self.unknown4 = "180"
        self.analyzerMode = ""
        self.resolution = ""
        self.magnification = "1"
        self.workFunction = ""
        self.targetBias = "0"
        self.analyzerWidthX = "0"
        self.analyzerWidthY = "0"
        self.analyzerTakeOffPolarAngle = "0"
        self.analyzerAzimuth = "0"
        self.speciesLabel = ""
        self.transitionLabel = ""
        self.particleCharge = "-1"
        self.abscissaLabel = "kinetic energy"
        self.abscissaUnits = "eV"
        self.abscissaStart = ""
        self.abscissaStep = ""
        self.noVariables = "2"
        self.variableLabel1 = "counts"
        self.variableUnits1 = "d"
        self.variableLabel2 = "Transmission"
        self.variableUnits2 = "d"
        self.signalMode = "pulse counting"
        self.dwellTime = ""
        self.noScans = ""
        self.timeCorrection = "0"
        self.sampleAngleTilt = "0"
        self.sampleTiltAzimuth = "0"
        self.sampleRotation = "0"
        self.noAdditionalParams = "2"
        self.paramLabel1 = "ESCAPE DEPTH TYPE"
        self.paramUnit1 = "d"
        self.paramValue1 = "0"
        self.paramLabel2 = "MFP Exponent"
        self.paramUnit2 = "d"
        self.paramValue2 = "0"
        self.numOrdValues = ""
        self.minOrdValue1 = ""
        self.maxOrdValue1 = ""
        self.minOrdValue2 = ""
        self.maxOrdValue2 = ""
        self.dataString = ""
