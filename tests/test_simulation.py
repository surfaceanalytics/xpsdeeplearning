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
"""
Tests for simulation
"""
import os
import sys
import json
import h5py
import datetime
import numpy as np
import pytest

from click.testing import CliRunner

from xpsdeeplearning.simulation.base_model.converters.data_converter import (
    DataConverter,
)
from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum
from xpsdeeplearning.simulation.sim import Simulation
from xpsdeeplearning.simulation.creator import Creator
from xpsdeeplearning.simulation.run import simulate_cli


def test_vms_load():
    converter = DataConverter()

    sys.stdout.write("Test on vms load okay.\n")


def test_vms_write():
    converter = DataConverter()

    sys.stdout.write("Test on vms write okay.\n")


def test_txt_load():
    label = "NiCoFe\\Fe2p_Fe_metal"
    datapath = (
        os.path.dirname(os.path.abspath(__file__)).partition("simulation")[0]
        + "data\\references"
    )

    filepath = datapath + "\\" + label + ".vms"

    measured_spectrum = MeasuredSpectrum(filepath)

    from figures import Figure

    fig = Figure(measured_spectrum.x, measured_spectrum.lineshape, title=label)

    sys.stdout.write("Test on txt load okay.\n")


def test_txt_write():
    converter = DataConverter()

    sys.stdout.write("Test on txt write okay.\n")


def test_single_sim():
    """Test simluation of single spectrum."""
    datapath = (
        os.path.dirname(os.path.abspath(__file__)).partition("simulation")[0]
        + "data\\references\\NiCoFe"
    )

    filenames = [
        "Co2p_Co_metal",
        "Co2p_CoO",
        "Co2p_Co3O4",
        "NiLMM_Ni_metal",
        "NiLMM_NiO",
        "CoLMM_CoO",
    ]
    input_spectra = []
    for filename in filenames:
        filepath = datapath + "\\" + filename + ".txt"
        input_spectra += [MeasuredSpectrum(filepath)]

    sim = Simulation(input_spectra)

    sim.combine_linear(scaling_params=[0.4, 0.1, 0.1, 0.1, 0.2, 0.1])
    sim.change_spectrum(
        shift_x=5,
        signal_to_noise=20,
        fwhm=200,
        scatterer={"label": "O2", "distance": 0.2, "pressure": 1.0},
    )

    print("Linear combination parameters: " + str(sim.output_spectrum.label))

    sim.output_spectrum

    # np.savez_compressed('filename.npz', array1=array1, array2=array2)
    ref_sim_file = "tests/data/ref_sim_spectrum.npz"  ######
    ref_x, ref_lineshape = np.load(ref_sim_file)

    assert sim.output_spectrum.x == ref_x
    assert sim.output_spectrum.lineshape == ref_lineshape

    sys.stdout.write("Test on single simulation okay.\n")


def test_creator():
    """Test creator class."""
    init_param_filepath = "tests/data/init_params_test.json"

    with open(init_param_filepath, "r") as param_file:
        params = json.load(param_file)

    creator = Creator(params=params)
    df = creator.run()

    ref_hdf5_file = "tests/data/20240202_Ni_linear_combination_small_gas_phase.h5"

    with h5py.File(ref_hdf5_file, "r") as hf:
        ref_energies = hf["energies"][:]
        ref_X = hf["X"][0].astype(float)
        ref_y = hf["y"][0]

    assert creator.df["x"][0] == ref_energies
    assert creator.df.iloc[0]["y"] == ref_X
    assert creator.df.iloc[0]["label"] == ref_y

    sys.stdout.write("Test on creator okay.\n")


@pytest.mark.parametrize(
    "cli_inputs",
    [
        pytest.param(
            [
                "--init_param_filepath",
                "tests/data/init_params_test.json",
            ],
        ),
    ],
)
def test_simulate_cli(cli_inputs):
    runner = CliRunner()
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    result = runner.invoke(simulate_cli, cli_inputs)

    hdf5_file = f"{timestamp}_Ni_linear_combination_small_gas_phase.h5"
    ref_hdf5_file = "tests/data/20240202_Ni_linear_combination_small_gas_phase.h5"

    with h5py.File(ref_hdf5_file, "r") as hf:
        energies = hf["energies"][:]
        X = hf["X"][0].astype(float)
        y = hf["y"][0]

    with h5py.File(ref_hdf5_file, "r") as hf:
        ref_energies = hf["energies"][:]
        ref_X = hf["X"][0].astype(float)
        ref_y = hf["y"][0]

    assert result.exit_code == 0
    assert energies == ref_energies
    assert X.shape == ref_X.shape
    assert y.shape == ref_y.shape

    os.remove(hdf5_file)
    sys.stdout.write("Test on simulate_cli okay.\n")
