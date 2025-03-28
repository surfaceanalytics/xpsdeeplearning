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

from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum
from xpsdeeplearning.simulation.creator import Creator
from xpsdeeplearning.simulation.run import simulate_cli
from xpsdeeplearning.simulation.sim import Simulation


def test_vms_load():
    """Test for loading VAMAS data."""
    filepath = "tests/data/sim/vms_test.vms"

    measured_spectrum = MeasuredSpectrum(filepath)
    assert "Fe2p zero line" in measured_spectrum.label
    assert measured_spectrum.spectrum_type == "core_level"
    assert measured_spectrum.x.shape == (1121,)
    assert measured_spectrum.lineshape.shape == (1121,)

    sys.stdout.write("Test on vms load okay.\n")


def test_txt_load():
    """Test for loading exported data in txt format."""
    filepath = "tests/data/sim/vms_export.txt"

    measured_spectrum = MeasuredSpectrum(filepath)
    assert "Fe2p all_zero" in measured_spectrum.label
    assert measured_spectrum.spectrum_type == "core_level"
    assert measured_spectrum.x.shape == (1121,)
    assert measured_spectrum.lineshape.shape == (1121,)
    sys.stdout.write("Test on txt load okay.\n")


def test_single_sim():
    """Test simluation of single spectrum."""
    datapath = "tests/data/sim/"

    filenames = ["Ni2p_Ni_metal.txt", "Ni2p_NiO.txt"]
    input_spectra = []
    for filename in filenames:
        filepath = os.path.join(datapath, filename)
        input_spectra += [MeasuredSpectrum(filepath)]

    sim = Simulation(input_spectra)

    sim.combine_linear(scaling_params=[0.4, 0.6])
    sim.change_spectrum(
        shift_x=5,
        signal_to_noise=20,
        fwhm=200,
        scatterer={"label": "O2", "distance": 0.2, "pressure": 1.0},
    )

    ref_sim_file = "tests/data/sim/ref_sim_spectrum.npz"
    data = np.load(ref_sim_file)

    for ref, val in zip(data["lineshape"], sim.output_spectrum.lineshape):
        np.testing.assert_allclose(ref, val, rtol=0.05)

    sys.stdout.write("Test on single simulation okay.\n")


def test_creator():
    """Test creator class."""
    init_param_filepath = "tests/data/sim/init_params_test.json"

    with open(init_param_filepath, "r") as param_file:
        params = json.load(param_file)

    creator = Creator(params=params)
    _ = creator.run()

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
                "tests/data/sim/init_params_test.json",
            ],
        ),
    ],
)
def test_simulate_cli(cli_inputs):
    """Test CLI function for data set simulation."""
    runner = CliRunner()
    result = runner.invoke(simulate_cli, cli_inputs)
    assert result.exit_code == 2

    sys.stdout.write("Test on simulate_cli okay.\n")
