#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/14/2023 1:40 AM
# @Author  : xiaomanl
# @File    : base_model.py
# @Software: PyCharm

import logging

#from abc import ABCMeta, abstractmethod

class BaseModel():
    def __init__(self, configs, processed_features, processed_labels):
        self.configs = configs
        self.processed_features = processed_features
        self.processed_labels = processed_labels
        self.logger = logging.getLogger("BaseModel")
        self.is_save_model = self.configs["is_save_model"]
        self.output_path = self.configs["output_path"]
        self.model_dict = {}

    def build_model(self, *args, **kwargs):

        raise NotImplementedError("build_model_method must be implemented by a subclass")

    def train(self, *args, **kwargs):

        raise NotImplementedError("train_method must be implemented by a subclass")


    # @abc.abstractmethod
    def predict(self, *args, **kwargs):

        raise NotImplementedError("predict_method must be implemented by a subclass")

    def run(self, *args, **kwargs):

        raise NotImplementedError("run_method must be implemented by a subclass")




