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
    input_param_file: str,
    reload_from_previous_folder: str = None,
    plot: bool = True
    ):
    """ Create multiple sets of similar spectra with the same settings."""
    with open(input_param_file, "r") as param_file:
        params = json.load(param_file)
        params["init_param_filepath"] = init_param_filepath

    creator = Creator(params)

    if reload_from_previous_folder:
        pkl_filename = ""
        dataset_name = ""
        pkl_filepath = os.path.join(reload_from_previous_folder, pkl_filename)

        creator.df = pd.read_pickle(pkl_filepath)

        # Reload parameters .
        param_filename = dataset_name + "_metadata.json"
        param_filepath = os.path.join(reload_from_previous_folder, param_filename)

        with open(param_filepath, "r") as param_file:
            creator.params = json.load(param_file)

    t0_run = time()
    df = creator.run()
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
    "--input-param-file",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the input parameter file to read.",
)
@click.option(
    "--output",
    default="output.nxs",
    help="The path to the output NeXus file to be generated.",
)

def simulate_cli(
    input_param_file: str,
    reload_from_previous_folder: str,
):
    """The CLI entrypoint for the convert function"""
    try:
        simulate(input_param_file, reload_from_previous_folder)
    except TypeError as exc:
        sys.tracebacklimit = 0
        raise TypeError(
            (
                "Please make sure you have the  entries in your "
                "parameter file:\n\n# NeXusParser Parameter File - v0.0.1"
                "\n\ndataconverter:\n\treader: value\n\tnxdl: value\n\tin"
                "put-file: value"
            )
        ) from exc


if __name__ == "__main__":
    os.chdir(
        os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
    )
    # Change the following two lines according to your folder structure ###
    init_param_folder = r"C:/Users/pielsticker/Lukas/MPI-CEC/Projects/deepxps/xpsdeeplearning/xpsdeeplearning/simulation/params/"
    init_param_filename = "init_params_Co_core_small_gas_phase.json"
    init_param_filepath = os.path.join(init_param_folder, init_param_filename)
    simulate(init_param_filepath)