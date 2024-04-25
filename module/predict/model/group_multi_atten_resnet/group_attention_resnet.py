#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/22/2023 11:07 AM
# @Author  : xiaomanl
# @File    : fcn_vec.py
# @Software: PyCharm

import keras
from keras.models import Model
from keras.layers import Dense, Conv1D,BatchNormalization,Activation, Input, Flatten, concatenate
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, RootMeanSquaredError,MeanSquaredError,LogCoshError
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import multi_gpu_model
import keras.backend as K
from keras.backend import reshape
from keras.layers import Embedding
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import custom_object_scope
from module.inference.model.base_class import BaseModel
from module.inference.utils import  calculate_running_time, LogPrintCallback, CustomDataGenerator, Train_predict_compare, Self_Attention, MultiHeadAtten
import os
import logging
import pandas as pd
import yaml
import sys
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# tf.config.optimizer.set_jit(True)
# tf.test.is_built_with_cuda()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)


class GroupMultiAttenResNet(BaseModel):
    def __init__(self, configs, processed_features, processed_labels):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("GroupMultiAttenResNet")
        self.mode_config = self.configs["GroupMultiAttenResNet_config"]
        self.build = self.mode_config["build"]
        self.verbose = self.mode_config["verbose"]
        self.batch_size = self.mode_config["batch_size"]
        self.nb_epochs = self.mode_config["nb_epochs"]
        self.mem_processed_numa_features = processed_features[0][0]
        self.mem_processed_char_feature = processed_features[0][1]
        self.cpu_processed_numa_features = processed_features[1][0]
        self.cpu_processed_char_feature = processed_features[1][1]
        self.system_processed_numa_features = processed_features[2][0]
        self.system_processed_char_feature = processed_features[2][1]
        self.workload_processed_numa_features = processed_features[3][0]
        self.workload_processed_char_feature = processed_features[3][1]
        self.numa_features = processed_features[4][0]
        self.char_features = processed_features[4][1]
        self.model_save_label = "test"
        self.isplot = self.mode_config["ifplot"]
        self.label_name_list = self.configs["label_name_list"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col": self.true_col})
        #,MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, LogCoshError,MeanAbsoluteError
        self.metrics = [MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(), MeanAbsolutePercentageError(name = "my mpe"), MeanSquaredLogarithmicError(), LogCoshError()]

        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.select_model = self.configs["select_model"]
        self.workload_name = self.configs["workload_names"][0]
        self.workload_scale = configs["label_scale"][self.workload_name]
        self.config_save_path = self.configs["config_save_path"]
        self.output_path = configs["output_path"]
        self.patience = self.mode_config["lr_patience"]
        self.factor = float(self.mode_config["lr_factor"])
        self.cooldown = float(self.mode_config["lr_cooldown"])
        self.min_lr = float(self.mode_config["min_lr"])
        self.opm_init_lr = float(self.mode_config["opm_init_lr"])
        self.decay_rate = float(self.mode_config["decay_rate"])
        self.decay_steps = self.mode_config["decay_steps"]
        self.n_feature_maps = self.mode_config["n_feature_maps"]
        self.freeze_train_model = self.configs["freeze_train_model"] if "freeze_train_model" in self.configs else 0
        self.output_path = configs["output_path"]
        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs["model_history_path"]


    def NN_init(self):
        tf.keras.initializers.he_normal(seed=None)

    def build_model(self):

        file = glob.glob(self.model_path + '/*.hdf5')[0]
        #

        model = tf.keras.models.load_model(file, custom_objects={ "MultiHeadAtten": MultiHeadAtten})
        self.logger.info("Load MODEL from: {}".format(file))
        model.summary()
        return model



    def predict(self):
        model = self.build_model()

        # input_x_data = (self.mem_char_x_test, self.mem_numer_x_test, self.cpu_char_x_test, self.cpu_numer_x_test,
        #                 self.system_char_x_test, self.system_numer_x_test, self.workload_char_x_test, self.workload_numer_x_test)
        input_x_data = (self.cpu_numer_x_test,  self.mem_numer_x_test,
                        self.system_char_x_test, self.system_numer_x_test, self.workload_char_x_test)
        # input_x_data = {"numer_input": self.numa_features, "char_input": self.char_features}
        input_x_data = {"mem_numer_input": self.mem_numer_x_test, "cpu_numer_input": self.cpu_numer_x_test,
                        "system_char_input": self.system_char_x_test,
                        "system_numer_input": self.system_numer_x_test,
                        "workload_char_input": self.workload_char_x_test}
        y_pred = model.predict(input_x_data)

        return y_pred

    @calculate_running_time
    def infer(self, train_with_all_data=False, result=None):
        self.logger.debug("Begin infer the model")

        self.logger.warning("Train model with all the data, without validation!")
        self.mem_char_x_test = self.data_reshape(self.mem_processed_char_feature)
        self.mem_numer_x_test = self.data_reshape(self.mem_processed_numa_features)
        self.cpu_char_x_test = self.data_reshape(self.cpu_processed_char_feature)
        self.cpu_numer_x_test = self.data_reshape(self.cpu_processed_numa_features)
        self.system_char_x_test = self.data_reshape(self.system_processed_char_feature)
        self.system_numer_x_test = self.data_reshape(self.system_processed_numa_features)
        self.workload_char_x_test = self.data_reshape(self.workload_processed_char_feature)
        self.workload_numer_x_test = self.data_reshape(self.workload_processed_numa_features)
        self.numa_features = self.data_reshape(self.numa_features)
        self.char_features = self.data_reshape(self.char_features)

        self.y_test = self.data_reshape(self.processed_labels)

        self.mem_char_x_train_shape = self.mem_char_x_test.shape[1:]
        self.cpu_char_x_train_shape = self.cpu_char_x_test.shape[1:]
        self.system_char_x_train_shape = self.system_char_x_test.shape[1:]
        self.workload_char_x_train_shape = self.workload_char_x_test.shape[1:]
        self.mem_numer_x_train_shape = self.mem_numer_x_test.shape[1:]
        self.cpu_numer_x_train_shape = self.cpu_numer_x_test.shape[1:]
        self.system_numer_x_train_shape = self.system_numer_x_test.shape[1:]
        self.workload_numer_x_train_shape = self.workload_numer_x_test.shape[1:]

        y_pred = self.predict()
        # keras.backend.clear_session()
        result = self.evaluate(self.y_test, y_pred)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, y_pred, self.y_test, save_path)


        return result


    def data_reshape(self, data):
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        return data_reshape


    def evaluate(self, y_test, y_predict):

        results = pd.DataFrame()
        # Calculate the Absolute Error (AE) for dataframe
        cols = self.label_name_list
        true_name = self.true_col
        predict_name = self.predict_col

        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
        if len(y_predict.shape) == 3:
            y_predict = np.reshape(y_predict, (y_predict.shape[0], y_predict.shape[1]))

        ae = np.abs(y_test - y_predict)
        se = (y_test - y_predict) ** 2
        ape = np.abs(y_test - y_predict) / y_test * 100

        for i, col in enumerate(cols):
            y_test_col = f'{true_name[i]}'
            y_predict_col = f'{predict_name[i]}'
            ae_col = "AE_" + col
            se_col = "SE_" + col
            ape_col = "APE(%)_" + col
            result = pd.DataFrame(
                {y_test_col: y_test[:, i], y_predict_col: y_predict[:, i], ae_col: ae[:, i], se_col: se[:, i],
                 ape_col: ape[:, i]})
            results = pd.concat([results, result], axis=1)

            mae = np.mean(ae[:, i])
            mse = np.mean(se[:, i])
            mape = np.mean(ape[:, i])
            p50_ape = round(np.quantile(ape[:, i], 0.5), 4)
            p90_ape = round(np.quantile(ape[:, i], 0.9), 4)
            p95_ape = round(np.quantile(ape[:, i], 0.95), 4)
            p99_ape = round(np.quantile(ape[:, i], 0.99), 4)
            max_ape = round(np.max(ape[:, i]), 4)

            count_3 = np.sum(ape[:, i] < 3)
            proportion_3 = round(count_3 / len(ape) * 100, 5)

            count_5 = np.sum(ape[:, i] < 5)
            proportion_5 = round(count_5 / len(ape) * 100, 4)

            count_10 = np.sum(ape[:, i] < 10)
            proportion_10 = round(count_10 / len(ape) * 100, 4)

            self.logger.info(f"------------------------------overall metrics snippet of {col}----------------------------------")
            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"MSE: {mse:.4f}")
            self.logger.info(f"MAPE(%): {mape:.4f}%")
            self.logger.info(f"Accuracy (APE < 3%) for {col}: {proportion_3}%")
            self.logger.info(f"Accuracy (APE < 5%) for {col}: {proportion_5}%")
            self.logger.info(f"Accuracy (APE < 10%) for {col}: {proportion_10}%")
            self.logger.info(f"P90 APE(%) for {col}: {p90_ape}%")
            self.logger.info(f"P95 APE(%) for {col}: {p95_ape}%")
            self.logger.info(f"P99 APE(%) for {col}: {p99_ape}%")
            self.logger.info(f"MAX APE(%) for {col}: {max_ape}%")

        return results

    def customed_mse(self, y_true, y_pred):
        weights = K.square(y_pred - y_true)
        squared_difference = K.square(y_pred - y_true)
        weighted_squared_difference = squared_difference * weights
        return K.mean(weighted_squared_difference, axis=-1)

    def plot_epochs_metric(self, hist, file_name, metric="loss"):
        try:
            history_dict = hist.history
            epochs = range(1, self.nb_epochs + 1)
            loss_values = history_dict[metric]
            plt.plot(epochs, loss_values, 'bo', label='Training ' + metric)
            if 'val_'+ metric in history_dict:
                val_loss_values = history_dict['val_'+ metric]
                plt.plot(epochs, val_loss_values, 'b', label='Validation '+metric)
            plt.title('Training and validation ' + metric)
            plt.ylabel(metric, fontsize='large')
            plt.xlabel('Epoch', fontsize='large')
            plt.legend(['train', 'val'], loc='upper right')
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error plot metric: {e}")






