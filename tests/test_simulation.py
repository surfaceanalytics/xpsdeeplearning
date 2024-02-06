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
import datetime
import json
import os
import sys

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from xpsdeeplearning.simulation.base_model.converters.data_converter import \
    DataConverter
from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum
from xpsdeeplearning.simulation.creator import Creator
from xpsdeeplearning.simulation.run import simulate_cli
from xpsdeeplearning.simulation.sim import Simulation


def test_vms_load():
    filepath = "tests/data/vms_test.vms"

    measured_spectrum = MeasuredSpectrum(filepath)
    assert ("Fe2p zero line" in measured_spectrum.label)
    assert measured_spectrum.spectrum_type == "core_level"
    assert measured_spectrum.x.shape == (1121,)
    assert measured_spectrum.lineshape.shape == (1121,)

    sys.stdout.write("Test on vms load okay.\n")

def test_txt_load():
    filepath = "tests/data/vms_export.txt"

    measured_spectrum = MeasuredSpectrum(filepath)
    assert ("Fe2p all_zero" in measured_spectrum.label)
    assert measured_spectrum.spectrum_type == "core_level"
    assert measured_spectrum.x.shape == (1121,)
    assert measured_spectrum.lineshape.shape == (1121,)
    sys.stdout.write("Test on txt load okay.\n")

def test_single_sim():
    """Test simluation of single spectrum."""
    datapath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)).partition("xpsdeeplearning")[0],
        "xpsdeeplearning", "xpsdeeplearning", "data", "references"
    )

    filenames = [
        "Ni2p_Ni_metal.txt",
        "Ni2p_NiO.txt"
    ]
    input_spectra = []
    for filename in filenames:
        filepath = os.path.join(datapath, filename)
        print(filepath)
        input_spectra += [MeasuredSpectrum(filepath)]

    sim = Simulation(input_spectra)

    sim.combine_linear(scaling_params=[0.4, 0.6])
    sim.change_spectrum(
        shift_x=5,
        signal_to_noise=20,
        fwhm=200,
        scatterer={"label": "O2", "distance": 0.2, "pressure": 1.0},
    )

    np.savez_compressed(
        "tests/data/ref_sim_spectrum.npz",
        energy=sim.output_spectrum.x,
        lineshape=sim.output_spectrum.lineshape
    )
    ref_sim_file = "tests/data/ref_sim_spectrum.npz"  ######
    data = np.load(ref_sim_file)

    assert (sim.output_spectrum.x == data["energy"]).all()
    assert (sim.output_spectrum.lineshape == data["lineshape"]).all()

    sys.stdout.write("Test on single simulation okay.\n")

def test_creator():
    """Test creator class."""
    init_param_filepath = "tests/data/init_params_test.json"

    with open(init_param_filepath, "r") as param_file:
        params = json.load(param_file)

    creator = Creator(params=params)
    df = creator.run()

    ref_hdf5_file = "tests/data/20240206_Ni_linear_combination_small_gas_phase.h5"

    with h5py.File(ref_hdf5_file, "r") as hf:
        ref_energies = hf["energies"][:]
        ref_X = hf["X"][0].astype(float)
        try:
            labels = [label.decode("utf-8") for label in hf["labels"][:]]
        except AttributeError:
            labels = [str(label) for label in hf["labels"][:]]

    assert (creator.df["x"][0] == ref_energies).all()
    assert creator.df.iloc[0]["y"].shape == ref_X.shape
    assert list(creator.df.iloc[0]["label"].keys()) == labels

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

    hdf5_file = "Ni_linear_combination_small_gas_phase.h5"
    ref_hdf5_file = "tests/data/20240206_Ni_linear_combination_small_gas_phase.h5"

    with h5py.File(ref_hdf5_file, "r") as hf:
        energies = hf["energies"][:]
        X = hf["X"][0].astype(float)
        y = hf["y"][0]

    with h5py.File(ref_hdf5_file, "r") as hf:
        ref_energies = hf["energies"][:]
        ref_X = hf["X"][0].astype(float)
        ref_y = hf["y"][0]

    assert result.exit_code == 2
    assert (energies==ref_energies).all()
    assert X.shape == ref_X.shape
    assert y.shape == ref_y.shape

    #os.remove(hdf5_file)
    sys.stdout.write("Test on simulate_cli okay.\n")