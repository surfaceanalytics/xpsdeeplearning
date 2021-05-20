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
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path.split(".", 1)[0]

        self.parent_path, _, self.model_name = self.model_path.rsplit("\\", 2)

    def _get_train_params(self):
        log_path = os.path.join(
            *[
                self.parent_path,
                "logs",
                self.model_name,
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
        dataset_name = self.dataset_path.split("\\", 1)[0]

        dataparams_path = os.path.join(
            *[
                os.path.dirname(dataset_path).partition("datasets")[0],
                dataset_name,
                "run_params.json",
            ]
        )

        with open(dataparams_path, "r") as param_file:
            data_params = json.load(param_file)

        for key in [
            "single",
            "variable_no_of_inputs",
            "timestamp",
            "no_of_simulations",
            "labels",
            "run_name",
            "output_datafolder",
            "no_of_files",
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

        return self.upload_params

    def _save_upload_params(self):
        upload_param_file = os.path.join(
            self.model_path, "upload_params.json"
        )

        with open(upload_param_file, "w", encoding="utf-8") as json_file:
            json.dump(
                self.upload_params, json_file, ensure_ascii=False, indent=4
            )

    def prepare_upload_folder(self, figure=True):
        self._save_upload_params()

        if figure:
            old_figure_path = os.path.join(
                *[self.parent_path, "figures", self.model_name, "model.png"]
            )
            new_figure_path = os.path.join(*[self.model_path, "model.png"])

            shutil.copyfile(old_figure_path, new_figure_path)

    def zip_all(self):
        shutil.make_archive(
            self.model_path, "zip", self.model_path,
        )


#%%
if __name__ == "__main__":
    model_path = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\saved_models\20210318_22h48m_Fe_unscattered_4_classes_linear_comb_small_gas_phase"
    dataset_path = r"C:\Users\pielsticker\Simulations\20210506_Fe_linear_combination_small_gas_phase\run_params.json"
    uploader = Uploader(model_path, dataset_path)
    uploader.prepare_upload_params()
    uploader.prepare_upload_folder(figure=True)
    uploader.zip_all()
