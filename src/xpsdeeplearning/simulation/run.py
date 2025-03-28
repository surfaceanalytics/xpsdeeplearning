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
This script is used to simulate many spectra using the Creator class
and save the data to a HDF5 file, along with the metadata in a JSON
file.
"""

from time import time
import os
import sys
import json
import pandas as pd
import click

from xpsdeeplearning.simulation.creator import (
    Creator,
    FileWriter,
    calculate_runtime,
)


def simulate(
    param_file: str, reload_from_previous_folder: str = None, plot: bool = True
):
    """Create multiple sets of similar spectra with the same settings."""
    with open(param_file, "r") as file:
        params = json.load(file)
        params["init_param_filepath"] = param_file

    creator = Creator(params)

    if reload_from_previous_folder:
        pkl_filename = ""
        dataset_name = ""
        pkl_filepath = os.path.join(reload_from_previous_folder, pkl_filename)

        creator.df = pd.read_pickle(pkl_filepath)

        # Reload parameters .
        param_filename = dataset_name + "_metadata.json"
        param_filepath = os.path.join(reload_from_previous_folder, param_filename)

        with open(param_filepath, "r") as param_file_reload:  # type: ignore
            creator.params = json.load(param_file_reload)  # type: ignore

    t0_run = time()
    _ = creator.run()
    t1_run = time()

    if plot:
        creator.plot_random(10)

    t0_save = time()
    writer = FileWriter(creator.df, creator.params)
    writer.to_file(filetypes=["pickle"], metadata=False)
    writer.to_file(filetypes=["hdf5"], metadata=True)
    t1_save = time()

    os.remove(writer.pkl_filepath)
    print(f"Runtime: {calculate_runtime(t0_run, t1_run)}.")
    print(f"HDF5 save runtime: {calculate_runtime(t0_save, t1_save)}.")


@click.command()
@click.option(
    "--param-file",
    default=None,
    required=True,
    help="The path to the input parameter file to read.",
)
@click.option(
    "--reload-from-previous-folder",
    default=None,
    help="The path to a previous run which is to be continued.",
)
def simulate_cli(
    param_file: str,
    reload_from_previous_folder: str,
):
    """The CLI entrypoint for the convert function"""
    try:
        simulate(param_file, reload_from_previous_folder, plot=False)
    except KeyError as exc:
        sys.tracebacklimit = 0
        raise KeyError(
            (
                "Please make sure you have these entries in your "
                "parameter file:\n"
                "output_datafolder: str\n"
                "name: str\n"
                "labels: list\\n"
                "spectra: list\n"
                "input_filenames: Dict\n"
                "no_of_simulations: int\n"
                "single: bool\n"
                "variable_no_of_inputs: bool\n"
                "always_auger: bool\n"
                "always_core: bool\n"
                "same_auger_core_percentage: bool\n"
                "ensure_same_length: bool\n"
                "eV_window: int\n"
                "normalize_inputs: bool\n"
                "normalize_outputs: bool\n"
                "broaden: bool\n"
                "shift_x: bool\n"
                "noise: bool\n"
                "scatter: bool\n"
                "sim_ranges: Dict with the keys\n"
                "\t shift_x: List[float]\n"
                "\t noise: List[int]\n"
                "\t FWHM: List[float]\n"
                "\t scatterers: Dict[int, str]\n"
                "\t pressure: List[float]\n"
                "\t distance: List[float]\n"
            )
        ) from exc


if __name__ == "__main__":
    working_dir = os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
    os.chdir(working_dir)
    # Change the following two lines according to your folder structure ###
    init_param_folder = r"C:/Users/pielsticker/Lukas/MPI-CEC/Projects/deepxps/xpsdeeplearning/xpsdeeplearning/simulation/params/"
    init_param_filename = "init_params_Co_core_small_gas_phase.json"
    init_param_filepath = os.path.join(init_param_folder, init_param_filename)
    simulate(init_param_filepath)
