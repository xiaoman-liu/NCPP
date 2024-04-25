#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/14/2023 1:40 AM
# @Author  : xiaomanl
# @File    : base_model.py
# @Software: PyCharm

import logging
import pandas as pd
import numpy as np
from src.train import model
import shutil

#from abc import ABCMeta, abstractmethod

class BaseModel():
    def __init__(self, configs, processed_features, processed_labels, train_indices, test_indices):
        self.configs = configs
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.processed_features = processed_features
        self.processed_labels = processed_labels
        self.logger = logging.getLogger("BaseModel")
        self.is_save_model = self.configs["is_save_model"]
        self.output_path = self.configs["output_path"]
        self.model_dict = {}

        self.model_history_path = self.configs["model_history_path"]
        self.copy_train_model = self.configs["copy_train_model"]
        self.train_with_all_data = self.configs["train_with_all_data"]
        self.train_label = self.configs["train_label"]

    def build_model(self, *args, **kwargs):

        raise NotImplementedError("build_model_method must be implemented by a subclass")

    def train(self, *args, **kwargs):

        raise NotImplementedError("train_method must be implemented by a subclass")


    # @abc.abstractmethod
    def predict(self, *args, **kwargs):

        raise NotImplementedError("predict_method must be implemented by a subclass")

    def run(self, *args, **kwargs):

        raise NotImplementedError("run_method must be implemented by a subclass")

    def copy_model(self):
        if self.copy_train_model and self.train_with_all_data:
            # new_model_path = self.output_path + "/model"
            new_model_path = self.output_path + "/" + self.train_label
            obj_path = self.model_history_path
            self.logger.info("start change model name to: {}".format(new_model_path))
            shutil.move(self.output_path + "/model", new_model_path)
            try:
                # self.logger.info("start copy weights to: {}".format(new_model_path+ "/../weights"))
                # shutil.copytree(new_model_path+ "/../weights", new_model_path)
                self.logger.info("start copy model to: {}".format(obj_path))
                shutil.copytree(new_model_path, obj_path)
                self.logger.info("finish copy model to: {}".format(obj_path))
            except FileExistsError:
                self.logger.warning("Destination folder already exists. Do you want to overwrite it? (Y/N)")
                choice = input().lower()
                if choice == 'y':
                    shutil.rmtree(obj_path)
                    shutil.copytree(new_model_path, obj_path)
                    self.logger.info("finish copy model from {}  \nto {}".format(new_model_path, obj_path))
                else:
                    self.logger.info("Not copy model from {} \n to: {}".format(new_model_path, obj_path))

        else:
            self.logger.info("Not copy model from {} \n to {}".format(self.output_path + "/model", self.model_history_path))




