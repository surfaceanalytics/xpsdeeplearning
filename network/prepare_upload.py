# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:05:50 2021

@author: pielsticker
"""
import os
import json
import shutil

#%%
class Uploader:
    def __init__(self, model_path, dataset_metadata_path):
        self.model_path = model_path
        self.dataset_metadata_path = dataset_metadata_path
        
    def _get_train_params(self):
        log_path = os.path.join(
            *[
                self.model_path,
                "logs",
                "hyperparameters.json",
            ]
        )

        with open(log_path, "r") as param_file:
            train_params = json.load(param_file)
            
        for key in [
            "input_filepath",
            "model_summary",
            "No. of training samples",
            "No. of validation samples",
            "No. of test samples",
            "Shape of each sample",
            "train_test_split",
            "train_val_split",
            "num_of_classes",
        ]:
            del train_params[key]

        return train_params

    def _get_data_params(self):
        with open(self.dataset_metadata_path, "r") as param_file:
            data_params = json.load(param_file)

        for key in [
            "single",
            "variable_no_of_inputs",
            "no_of_simulations",
            "labels",
            "output_datafolder",
        ]:
            del data_params[key]

        return data_params

    def prepare_upload_params(self,):
        train_params = self._get_train_params()
        data_params = self._get_data_params()

        name = train_params["time"] + "_" + train_params["exp_name"]
        labels = train_params["labels"]

        for key in ["time", "exp_name", "labels"]:
            del train_params[key]

        self.upload_params = {
            "name": name,
            "labels": labels,
            "dataset_params": data_params,
            "train_params": train_params,
        }

    def save_upload_params(self):
        upload_param_file = os.path.join(
            self.model_path, "upload_params.json"
        )

        with open(upload_param_file, "w", encoding="utf-8") as json_file:
            json.dump(
                self.upload_params, json_file, ensure_ascii=False, indent=4
            )
            
        print("JSON file was prepared for upload!")

    def zip_all(self):
        shutil.make_archive(
            self.model_path, "zip", self.model_path,
        )


#%%
if __name__ == "__main__":
    model_path = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs\20210531_16h27m_Co_3_classes_linear_comb_small_gas_phase"
    dataset_path = r"C:\Users\pielsticker\Simulations\20210528_Co_linear_combination_small_gas_phase_metadata.json"
    uploader = Uploader(model_path, dataset_path)
    uploader.prepare_upload_params()
    uploader.save_upload_params()
    #uploader.zip_all()
