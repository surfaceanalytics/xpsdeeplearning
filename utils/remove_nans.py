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
This script can be used to remove NaN values from a classifier training
history.
"""
import csv
import numpy as np
import pandas as pd


def get_total_history(csv_filepath):
    """Get training history from csv file."""
    history = {}
    try:
        with open(csv_filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, item in row.items():
                    if key not in history.keys():
                        history[key] = []
                    history[key].append(float(item))
    except FileNotFoundError:
        pass

    return history


def remove_nan_rows(history):
    """Remove NaN values from training history."""
    nan_indices = []

    for key, data_list in history.items():
        nan_indices.extend(np.where(np.isnan(data_list))[0])

    nan_indices = set(nan_indices)

    for key, data_list in history.items():
        history[key] = [
            data_point for i, data_point in enumerate(data_list) if i not in nan_indices
        ]

    return history


def write_new_history(history):
    """Write history back to new CSV file."""
    csv_file = r"C:\Users\pielsticker\Downloads\log_new.csv"

    pd.DataFrame(history).to_csv(csv_file, index=False)


if __name__ == "__main__":
    csv_file = r"C:\Users\pielsticker\Downloads\log.csv"

    history = get_total_history(csv_filepath=csv_file)
    history = remove_nan_rows(history)
    history["epoch"] = list(np.arange(0, len(history["loss"]), 1))

    write_new_history(history)
