#Copyright (C) <2024> Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.abspath(__file__))

import pandas as pd
from pathlib import Path
import logging

from module.train.train_utils import calculate_running_time, mkdir, save_data_encoder, read_config, set_logger
from module.train.generate_data import DataLoader
from module.train.group_embedding import  GroupFeatureEmbedding
from module.train.postprocessor import DataPostprocessor
from module.train.ncpp import GroupMultiAttenResNet
from module.train.train_utils.additional import merge_K_fold_results

class Trainer:
    def __init__(self, output_path="../../train_results", config_path="./config", module_path="./"):
        self.root_dir = Path(__file__).resolve().parent
        self.configs = read_config(self.root_dir, output_path, config_path, module_path)
        self.logger = self._set_logger()

        self.select_model = self.configs["select_model"]
        self.k_fold = True if self.configs["K_Fold"] and not self.configs["train_with_all_data"] else False
        self.train_with_all_data = True if self.configs["train_with_all_data"] else False
        self.model_dict = self.configs["model_dict"]
        self.n_split = self.configs["n_split"]
        self.load_data_method = globals()[self.configs[self.select_model + "_workflow_class"]["load_data"]]
        self.process_data_method = globals()[self.configs[self.select_model + "_workflow_class"]["process_data"]]
        self.split_folder = 'f"split_{i}_fold"'
        self.configs.update({"split_folder": self.split_folder})
        self.output_path = self.configs["output_path"]
        self.if_label_scale = self.configs["if_label_scale"]

        _, self.model_name = self.model_dict[self.select_model].rsplit('.', 1)
        self.model = globals()[self.model_name]
        # self.label_scaler = MinMaxScaler(feature_range=(0, 500))
        # self.workload_name = self.configs["workload_names"][0]
        # self.label_scale = self.configs["label_scale"].get(self.workload_name, 1)



    def _set_logger(self):
        logger = set_logger(self.configs)
        return logging.getLogger("TrainModule")


    def load_data(self):
        self.data_loader = self.load_data_method(self.configs, self.k_fold)
        self.filter_data, self.label, self.train_inds, self.test_inds, self.all_train_data, self.configs = self.data_loader.run()
        if self.train_with_all_data:
            self.train_inds = self.filter_data.index.values

    def preprocess_data(self):
        self.configs["filtered_columns"] = list(self.filter_data.columns)
        self.data_processor = self.process_data_method(self.configs)
        print(f"{self.data_processor}")
        # if self.train_with_all_data:
        #     save_data_encoder(self.output_path, self.data_processor)
        self.processed_feature, self.configs , self.processed_label = self.data_processor.run(self.filter_data, self.label)
        # if self.if_label_scale:
        #     label_temp = self.label.values
        #     label_temp = self.label_scaler.fit_transform(label_temp)
        #     self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        #     self.configs.update({"label_scaler": (self.label_scaler.min_, self.label_scaler.scale_)})
        # else:
        #     self.label /= self.label_scale

    def train_model(self, save_path):
        mkdir(save_path)

        self.model_module = self.model(self.configs, self.processed_feature, self.processed_label,
                                                          self.train_inds, self.test_inds, save_path)
        self.predicted_results, hist = self.model_module.run(self.train_with_all_data)
        return hist

    def postprecess_data(self, save_path=None, hist_all=None):
        # if self.if_label_scale:
        #     label_temp = self.label.values
        #     label_temp = self.label_scaler.inverse_transform(label_temp)
        #     self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        # else:
        #     self.label *= self.label_scale
        self.data_postprocessor = DataPostprocessor(save_path,self.configs, self.filter_data, self.processed_label, self.train_inds, self.test_inds, self.predicted_results, self.processed_feature, hist_all=hist_all)
        self.data_postprocessor.run(self.train_with_all_data)


    def train_with_Kfold(self):

        all_train_indices = self.train_inds
        all_test_indices = self.test_inds
        hist_all = []
        for i in range(1, self.n_split + 1):
            self.logger.info(f"-----------------Begin K Fold train and validation Split {i} ------------------")
            kth_fold_save_path = os.path.join(self.output_path,
                                                     eval(self.split_folder)).replace("\\", "/")

            # Process features
            self.train_inds = pd.Index(all_train_indices[i - 1])
            self.test_inds = pd.Index(all_test_indices[i - 1])
            self.configs.update({"k_fold_order": i})
            self.configs.update({"kth_fold_save_path": kth_fold_save_path})
            # self.preprocess_data()

            hist = self.train_model(kth_fold_save_path)
            hist_all.append(hist)
            self.postprecess_data(kth_fold_save_path)
            self.logger.info(f"Finish K Fold train and validation Split {i}\n")
        variation_data, test_results = merge_K_fold_results(self.configs)
        return hist_all


    @calculate_running_time
    def run(self):
        self.load_data()
        self.preprocess_data()
        if self.k_fold and not self.train_with_all_data:
            self.logger.info("")
            hist_all = self.train_with_Kfold()
        else:
            self.logger.info("-----------------Begin train with all the data without validation ------------------")
            self.logger.info("")
            hist_all = self.train_model(self.output_path)
        self.postprecess_data(hist_all=hist_all)


        return self.configs



if __name__ == "__main__":
    trainer = Trainer()
    configs = trainer.run()





