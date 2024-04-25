#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 4:49 PM
# @Author  : xiaomanl
# @File    : test.py
# @Software: PyCharm

import os
import sys

import pandas as pd
from pathlib import Path
import logging
import joblib

from module.inference.utils import calculate_running_time, mkdir, save_data_encoder, read_config, set_logger
from module.inference.generate_data import DataLoader
from module.inference.data_preprocess import FeatureEncoder, GroupFeatureEmbedding
from module.inference.data_postprocess import DataPostprocessor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from module.inference.utils.model_utils import MinMaxScaler, NorScaler, OneHotEncoder, TextTokenizer
from module.inference.utils.additional import merge_K_fold_results
from module.inference.utils import param_search, DatabaseManager
from module.inference.model import LinearModel, XgboostModel, FCN, FCN_Vec, ResNet, AttenResNet, MultiAttenResNet, RandomForest, SVMModel, LSTMModel, Ridge, GroupMultiAttenResNet



class Predictor:
    def __init__(self, output_path="../../infer_results", config_path="./"):
        self.root_dir = Path(__file__).resolve().parent
        self.configs = read_config(self.root_dir, config_path, output_path)
        self.logger = self._set_logger()
        self.select_model = self.configs["select_model"]
        self.output_path = self.configs["output_path"]
        # self.if_label_scale = self.configs["if_label_scale"]
        #
        # # self.label_scaler = MinMaxScaler()
        # self.workload_name = self.configs["workload_names"][0]
        # self.label_scale = self.configs["label_scale"].get(self.workload_name, 1)
        self.encoder_path = self.configs["encoder_path"]

        self.model_dict = self.configs["model_dict"]
        _, self.model_name = self.model_dict[self.select_model].rsplit('.', 1)
        self.model = globals()[self.model_name]
        self.infer_upload_sql = self.configs["infer_upload_sql"]
        self.if_label_scale = self.configs["if_label_scale"]

    def get_model(self):
        if self.select_model == "Ridge" or self.select_model == "Lasoo" or self.select_model == "ElasticNet":
            model =  globals()["LinearModel"]
        else:
            model = globals()[self.model_name]
        return model

    def _set_logger(self):
        logger = set_logger(self.configs)
        return logging.getLogger("InferModule")


    def load_data(self):
        self.data_loader = DataLoader(self.configs)
        self.filter_data, self.label,self.all_train_data = self.data_loader.run()

    def preprocess_data(self):
        self.configs["filtered_columns"] = list(self.filter_data.columns)
        if self.select_model == "GroupMultiAttenResNet":
            self.data_processor = GroupFeatureEmbedding(self.configs)
        else:
            self.data_processor = FeatureEncoder(self.configs)
        self.processed_feature, self.processsed_label = self.data_processor.run(self.filter_data, self.label)
        # if self.if_label_scale:
        #     label_temp = self.label.values
        #     label_temp = self.label_scaler.fit_transform(label_temp)
        #     self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        # else:
        #     self.label /= self.label_scale

    def only_predict(self, save_path):
        self.param_search = self.select_model == "XgboostModel" and self.configs["XgboostModel_config"]["param_search"]
        mkdir(save_path)

        self.model_module = self.model(self.configs, self.processed_feature, self.processsed_label)
        self.predicted_results = self.model_module.infer()

    def postprecess_data(self, save_path=None):
        # if self.if_label_scale:
        #     label_temp = self.label.values
        #     label_temp = self.label_scaler.inverse_transform(label_temp)
        #     self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        # else:
        #     self.label *= self.label_scale
        self.data_postprocessor = DataPostprocessor(save_path, self.configs, self.filter_data, self.processsed_label, self.predicted_results)
        self.all_validate_dataset = self.data_postprocessor.run()

    def upload_sql(self, variation_data="", test_results=""):
        uploader = DatabaseManager(self.configs)
        uploader.run("inference_result", self.all_validate_dataset)


    @calculate_running_time
    def run(self):
        self.load_data()
        self.preprocess_data()

        self.logger.info("Begin to infer")
        self.only_predict(self.output_path)
        self.postprecess_data()
        if self.infer_upload_sql:
            self.upload_sql()


        return self.configs



if __name__ == "__main__":
    Predictor = Predictor()
    configs = Predictor.run()



