# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:52:20 2021

@author: pielsticker
"""
import csv
import numpy as np
import pandas as pd


def get_total_history(filepath):
    history = {}
    try:
        with open(csv_file, newline="") as csvfile:
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
    nan_indices = []

    for key, data_list in history.items():
        nan_indices.extend(np.where(np.isnan(data_list))[0])

    nan_indices = set(nan_indices)

    for key, data_list in history.items():
        history[key] = [
            data_point
            for i, data_point in enumerate(data_list)
            if i not in nan_indices
        ]

    return history


def write_new_history(history):
    csv_file = r"C:\Users\pielsticker\Downloads\log_new.csv"

    pd.DataFrame(history).to_csv(csv_file, index=False)


csv_file = r"C:\Users\pielsticker\Downloads\log.csv"

history = get_total_history(filepath=csv_file)
history = remove_nan_rows(history)
history["epoch"] = list(np.arange(0, len(history["loss"]), 1))

write_new_history(history)
