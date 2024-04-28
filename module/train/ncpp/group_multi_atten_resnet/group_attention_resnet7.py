#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/22/2023 11:07 AM
# @Author  : xiaomanl
# @File    : fcn_vec.py
# @Software: PyCharm

import keras
from keras.models import Model
from keras.layers import Dense, Conv1D,Activation, Input, Flatten, concatenate, Dropout
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, RootMeanSquaredError,\
                            MeanSquaredError,LogCoshError, Accuracy
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import multi_gpu_model
import keras.backend as K
from keras.backend import reshape
from keras.layers import Embedding
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import custom_object_scope
from module.train.ncpp.base_class import BaseModel
from module.train.train_utils import  (calculate_running_time, LogPrintCallback, CustomDataGenerator, Train_predict_compare, Self_Attention,
                                    MultiHeadAtten, genereate_feature_list, mkdir)
import os
import logging
import pandas as pd
import yaml
import sys
from keras.layers import LayerNormalization as normalization_layer
from keras.utils import pad_sequences
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# tf.config.run_functions_eagerly(True)


# tf.config.optimizer.set_jit(True)
# tf.test.is_built_with_cuda()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)


class GroupMultiAttenResNet(BaseModel):
    def __init__(self, configs, processed_features, processed_labels, train_indices, test_indices, k_fold_save_path):
        super().__init__( configs, processed_features, processed_labels, train_indices, test_indices)
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
        self.model_save_label = "test"
        self.save_path = k_fold_save_path
        self.isplot = self.mode_config["ifplot"]
        self.label_name_list = self.configs["label_name_list"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col": self.true_col})
        #,MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, LogCoshError,MeanAbsoluteError
        self.metrics = [MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(), MeanAbsolutePercentageError(name = "my mpe"), MeanSquaredLogarithmicError(), LogCoshError(), Accuracy()]

        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.attention_model_path = self.mode_config["attention_model_path"]
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
        self.freeze_train_model = self.configs["freeze_train_model"]
        self.char_token_order = self.configs["char_token_order"]
        self.visualize_attention = self.mode_config["if_visual_atten"]
        self.numeric_token_order = self.configs["num_token_order"]
        self.report_feature_mapping = self.configs["report_feature_mapping"]
        self.pooled_char_feature = self.mode_config["pooled_char_feature"]
        self.multi_heads = self.mode_config["multi_heads"]
        self.kth_fold_save_path = self.configs["kth_fold_save_path"] if "kth_fold_save_path" in self.configs else ""
        self.final_path = self.output_path if self.train_with_all_data else self.kth_fold_save_path
        mkdir(self.final_path + "/ncpp")
        self.attention_model_path = self.mode_config["attention_model_path"]
        self.use_pre_train_attmodel = self.mode_config["use_pre_train_attmodel"]
        self.sample_index = self.mode_config["sample_index"]


    def NN_init(self):
        tf.keras.initializers.he_normal(seed=None)

    def resnet_block(self, concatenated_inputs, feature_maps, name_suffix):
        # BLOCK 1
        # attention_layer = Self_Attention(128)(concatenated_inputs)
        conv_x = keras.layers.Conv1D(filters=feature_maps, kernel_size=8, padding='same', name="fix_convx1" + name_suffix )(concatenated_inputs)

        # conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(concatenated_inputs)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=feature_maps, kernel_size=5, padding='same',name="fix_conv2" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=feature_maps, kernel_size=3, padding='same', name="fix4" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=feature_maps, kernel_size=1, padding='same', name="fix6" + name_suffix)(
            concatenated_inputs)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=8, padding='same', name="fix9" + name_suffix )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=5, padding='same', name="fix10" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=3, padding='same', name="fix11" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=1, padding='same', name="fix12" + name_suffix)(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        #
        # # # BLOCK 3
        # conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=8, padding='same', name="fix13" + name_suffix)(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=5, padding='same',name="fix14" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=feature_maps * 2, kernel_size=3, padding='same', name="fix15" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)
        # shortcut_y = keras.layers.BatchNormalization()(output_block_1)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])

        # output_block_3 = keras.layers.Dense(feature_maps, activation="relu", name="fix16" + name_suffix)(output_block_3)
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        #
        # # transformer encoder
        #
        # output_block_4 = MultiHeadAtten(output_dim=feature_maps, nheads=1, name="fix17" + name_suffix)(output_block_3)
        #
        # output_block_5 = keras.layers.add([output_block_3, output_block_4])
        # # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # # output_block_5 = keras.layers.add([output_block_2, output_block_4])
        # output_block_5 = keras.layers.BatchNormalization()(output_block_5)
        #
        # FFN_5 = keras.layers.Dense(feature_maps, activation='linear', name="fix18" + name_suffix)(output_block_5)
        # FFN_5 = keras.layers.add([output_block_5, FFN_5])
        # FFN_5 = keras.layers.BatchNormalization()(FFN_5)

        return output_block_3

    def d2resnet_block(self, concatenated_inputs, feature_maps, name_suffix):
        # BLOCK 1
        # attention_layer = Self_Attention(128)(concatenated_inputs)
        conv_x = keras.layers.Conv2D(filters=feature_maps, kernel_size=8, padding='same', name="fix_convx1" + name_suffix )(concatenated_inputs)

        # conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(concatenated_inputs)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=feature_maps, kernel_size=5, padding='same',name="fix_conv2" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=feature_maps, kernel_size=3, padding='same', name="fix4" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=feature_maps, kernel_size=1, padding='same', name="fix6" + name_suffix)(
            concatenated_inputs)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=8, padding='same', name="fix9" + name_suffix )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=5, padding='same', name="fix10" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=3, padding='same', name="fix11" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=1, padding='same', name="fix12" + name_suffix)(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        #
        # # # BLOCK 3
        # conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=8, padding='same', name="fix13" + name_suffix)(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=5, padding='same',name="fix14" + name_suffix)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=3, padding='same', name="fix15" + name_suffix)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)
        # shortcut_y = keras.layers.BatchNormalization()(output_block_1)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])

        # output_block_3 = keras.layers.Dense(feature_maps, activation="relu", name="fix16" + name_suffix)(output_block_3)
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        output_block_4 = keras.layers.AveragePooling2D(pool_size=(1,4), strides=None, padding="valid")(output_block_3)
        output_block_4 = K.squeeze(output_block_4, axis=2)
        if self.pooled_char_feature:
            output_block_4 = self.pooled_feature(output_block_4, name_suffix)

        #
        # # transformer encoder
        #
        # output_block_4 = MultiHeadAtten(output_dim=feature_maps, nheads=1, name="fix17" + name_suffix)(output_block_3)
        #
        # output_block_5 = keras.layers.add([output_block_3, output_block_4])
        # # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # # output_block_5 = keras.layers.add([output_block_2, output_block_4])
        # output_block_5 = keras.layers.BatchNormalization()(output_block_5)
        #
        # FFN_5 = keras.layers.Dense(feature_maps, activation='linear', name="fix18" + name_suffix)(output_block_5)
        # FFN_5 = keras.layers.add([output_block_5, FFN_5])
        # FFN_5 = keras.layers.BatchNormalization()(FFN_5)

        return output_block_4

    def pooled_feature(self, output_block_4, name_suffix):
        pooled_features = []
        left = 0
        for i in range(len(self.char_token_order[name_suffix])):
            dim = self.char_token_order[name_suffix][i][1]
            right = dim + left
            feature_output = output_block_4[:, left:right, :]
            feature_output = keras.layers.AveragePooling1D(pool_size=dim, strides=None, padding="valid")(feature_output)
            pooled_features.append(feature_output)
            left = right
        if len(pooled_features) == 1:
            return pooled_features[0]
        concatenated_features = keras.layers.concatenate(pooled_features, axis=1)
        return concatenated_features
    def embedding(self, char_input, embeding_dim=4):
        char_embedding = keras.layers.Masking(mask_value=0)(char_input)
        char_embedding = Embedding(input_dim=100, output_dim=embeding_dim, input_length=1)(char_embedding)
        char_embedding_flatten = Flatten()(char_embedding)
        batch_size, a = char_embedding_flatten.shape
        char_embedding_final = reshape(char_embedding_flatten, (-1, a, 1))
        return char_embedding_final

    def d2embedding(self, char_input, embeding_dim=4):
        char_embedding = keras.layers.Masking(mask_value=0)(char_input)
        char_embedding = Embedding(input_dim=100, output_dim=embeding_dim, input_length=1)(char_embedding)
        per_char_embedding = K.permute_dimensions(char_embedding, (0, 1, 3, 2))
        return per_char_embedding

    def padding_and_masking(self, input, max_length):
        origin_shape = input.shape[2]
        if origin_shape < max_length:
            pad_shape = max_length - origin_shape
            # padded_input = ZeroPadding1D(padding=(0, 0,pad_shape))(input)
            # pad = tf.constant(-0.111, shape=[tf.shape(input)[0], tf.shape(input)[1], pad_shape])
            # padded_input = tf.concat([input, pad], axis=2)
            padding_input = tf.pad(input, paddings=[[0, 0], [0, 0], [0, pad_shape]], constant_values=-0.111)
            masking_layer = tf.keras.layers.Masking(mask_value=-0.111)
            masked_input = masking_layer(padding_input)
            return masked_input
        else:
            return input




    def attention_block(self, input, heads=1, name = "cross_group"):
        output_block = keras.layers.LayerNormalization()(input)
        output_block, QK, or_QKo, Qi, x, w = MultiHeadAtten(output_dim=input.shape[2], nheads=heads, name=name)(output_block)
        # output_block = Dropout(0.1)(output_block)
        output_block = keras.layers.add([input, output_block])
        # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # output_block_5 = keras.layers.add([output_block_2, output_block_4])

        FFN_6 = keras.layers.LayerNormalization()(output_block)
        FFN_6 = keras.layers.Dense(FFN_6.shape[2], activation='relu')(output_block)
        FFN_6 = keras.layers.Dense(FFN_6.shape[2])(output_block)

        # FFN_6 = Dropout(0.1)(FFN_6)
        FFN_6 = keras.layers.add([output_block, FFN_6])

        # # 2 layer
        # output_block, QK = MultiHeadAtten(output_dim=input.shape[2], nheads=heads, name=name)(FFN_6)
        # output_block = keras.layers.add([input, output_block])
        # # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # # output_block_5 = keras.layers.add([output_block_2, output_block_4])
        # output_block = keras.layers.BatchNormalization()(output_block)
        #
        # FFN_6 = keras.layers.Dense(output_block.shape[2], activation='linear')(output_block)
        # FFN_6 = keras.layers.add([output_block, FFN_6])
        # FFN_6= keras.layers.BatchNormalization()(FFN_6)
        return FFN_6


    def build_model(self):

        mem_numeric_var_input = Input(shape=self.mem_numer_x_train_shape, name='mem_numer_input')


        cpu_numeric_var_input = Input(shape=self.cpu_numer_x_train_shape, name='cpu_numer_input')



        system_char_var_input = Input(shape=self.system_char_x_train_shape, name='system_char_input')
        system_char_embedding = self.d2embedding(system_char_var_input, embeding_dim=4)
        system_numeric_var_input = Input(shape=self.system_numer_x_train_shape, name='system_numer_input')
        # system_concatenated_inputs = concatenate([ system_numeric_var_input, system_char_embedding], axis=1)


        workload_char_var_input = Input(shape=self.workload_char_x_train_shape, name='workload_char_input')
        workload_char_embedding = self.d2embedding(workload_char_var_input)

        # char_concatenated_inputs = concatenate([system_char_embedding, workload_char_embedding], axis=1)
        # char_concatenated_inputs_shape = char_concatenated_inputs.shape[2]

        numer_concatenated_inputs = concatenate([mem_numeric_var_input, cpu_numeric_var_input, system_numeric_var_input], axis=1)
        numer_concatenated_inputs_shape = numer_concatenated_inputs.shape[2]
        # max_length = max(mem_numeric_var_input.shape[2], cpu_numeric_var_input.shape[2], system_concatenated_inputs.shape[2], workload_char_embedding.shape[2])

        # mem_numeric_var_input_padding = self.padding_and_masking(mem_numeric_var_input, max_length)
        # cpu_numeric_var_input_padding = self.padding_and_masking(cpu_numeric_var_input, max_length)
        # system_concatenated_inputs_padding = self.padding_and_masking(system_concatenated_inputs, max_length)
        # workload_char_embedding_padding = self.padding_and_masking(workload_char_embedding, max_length)
        #
        #
        #
        # raw_concatenated_inputs = concatenate([mem_numeric_var_input_padding, cpu_numeric_var_input_padding, system_concatenated_inputs_padding, workload_char_embedding_padding])
        # raw_concatenated_inputs = concatenate([numer_concatenated_inputs, char_concatenated_inputs], axis=1)
        system_char_feature_resnet_block = self.d2resnet_block(system_char_embedding, self.n_feature_maps, "System_info")
        workload_char_feature_resnet_block = self.d2resnet_block(workload_char_embedding, self.n_feature_maps, "Workload_info")
        char_feature_resnet_block = concatenate([system_char_feature_resnet_block, workload_char_feature_resnet_block], axis=1)

        # char_feature_resnet_block = self.d2resnet_block(char_concatenated_inputs, self.n_feature_maps, "all")

        num_feature_resnet_block = self.resnet_block(numer_concatenated_inputs, self.n_feature_maps, "num")
        part1 = num_feature_resnet_block[:, :self.mem_numer_x_train_shape[0], :]
        part2 = num_feature_resnet_block[:, self.mem_numer_x_train_shape[0]:self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0], :]
        part3 = num_feature_resnet_block[:, self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0]:self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0] + self.system_numer_x_train_shape[0], :]
        # char = all_feature_resnet_block[:, self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0] + self.system_numer_x_train_shape[0]:, :]
        char_features_attention_block = self.attention_block(char_feature_resnet_block, heads=self.multi_heads, name="char")

        part1_attention_block = self.attention_block(part1, heads=self.multi_heads, name="mem_num")
        part2_attention_block = self.attention_block(part2, heads=self.multi_heads, name="cpu_num")
        part3_attention_block = self.attention_block(part3, heads=self.multi_heads, name="system_num")
        # part4_attention_block = self.attention_block(part4, heads=2)
        part1_attention_block = keras.layers.AveragePooling1D(pool_size=part1_attention_block.shape[1], strides=None, padding="valid")(part1_attention_block)
        part2_attention_block = keras.layers.AveragePooling1D(pool_size=part2_attention_block.shape[1], strides=None, padding="valid")(part2_attention_block)
        part3_attention_block = keras.layers.AveragePooling1D(pool_size=part3_attention_block.shape[1], strides=None, padding="valid")(part3_attention_block)
        char_features_attention_block = keras.layers.AveragePooling1D(pool_size=char_feature_resnet_block.shape[1], strides=None, padding="valid")(char_features_attention_block)

        concatenated_inputs = concatenate([part1_attention_block, part2_attention_block, part3_attention_block, char_features_attention_block], axis=1)
        # concatenated_inputs = concatenate([all_feature_conv_attention_block, conv_attention_block_1, conv_attention_block_2, conv_attention_block_3, conv_attention_block_4])
        concatenated_inputs_shape = concatenated_inputs.shape[2]

        group_attention_block = self.attention_block(concatenated_inputs, heads=self.multi_heads, name="cross_group")
        concatenated_inputs_ave = keras.layers.AveragePooling1D(pool_size=concatenated_inputs.shape[1], strides=None,
                                                              padding="valid")(concatenated_inputs)

        # output_block_7, QK = MultiHeadAtten(output_dim=concatenated_inputs_shape, nheads=self.multi_heads, name="cross_group1")(concatenated_inputs)
        # output_block_7 = keras.layers.add([concatenated_inputs, output_block_7])
        # # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # # output_block_5 = keras.layers.add([output_block_2, output_block_4])
        # output_block_7 = keras.layers.BatchNormalization()(output_block_7)
        #
        # # FFN_6 = keras.layers.Dense(output_block_7.shape[2], activation='linear')(output_block_7)
        # # FFN_6 = keras.layers.add([output_block_7, FFN_6])
        # # FFN_6= keras.layers.BatchNormalization()(FFN_6)
        #
        # # output_block_7, QK = MultiHeadAtten(output_dim=concatenated_inputs_shape, nheads=self.multi_heads, name="cross_group")(FFN_6)
        # # output_block_7 = keras.layers.add([concatenated_inputs, output_block_7])
        # # # output_block_4 = MultiHeadAtten(output_dim=n_feature_maps * 2, nheads=2)(output_block_2)
        # # # output_block_5 = keras.layers.add([output_block_2, output_block_4])
        # # output_block_7 = keras.layers.BatchNormalization()(output_block_7)
        #
        # FFN_6 = keras.layers.Dense(output_block_7.shape[2], activation='linear')(output_block_7)
        # FFN_6 = keras.layers.add([output_block_7, FFN_6])
        # FFN_6 = keras.layers.BatchNormalization()(FFN_6)



        flatten = Flatten()(concatenated_inputs_ave)
        if not self.pooled_char_feature:
            flatten = keras.layers.Dense(64 * 2, activation='linear')(flatten)

        output_layer = keras.layers.Dense(len(self.label_name_list), activation='linear')(flatten)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=0.00001)
        model = Model(inputs={"mem_numer_input": mem_numeric_var_input, "cpu_numer_input": cpu_numeric_var_input,
                              "system_char_input": system_char_var_input,
                              "system_numer_input": system_numeric_var_input,
                              "workload_char_input": workload_char_var_input}, outputs=output_layer)
        # ncpp = Model(inputs={"mem_char_input": mem_char_var_input, "mem_numer_input": mem_numeric_var_input,
        #                       "cpu_char_input": cpu_char_var_input, "cpu_numer_input": cpu_numeric_var_input,
        #                       "system_char_input": system_char_var_input, "system_numer_input": system_numeric_var_input,
        #                         "workload_char_input": workload_char_var_input, "workload_numer_input": workload_numeric_var_input}, outputs=output_layer)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.opm_init_lr, decay_steps=self.decay_steps, decay_rate=self.decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1), metrics=self.metrics)# , run_eagerly=True

        # ncpp.summary()

        return model, lr_schedule


    def train(self):
        # if not tf.test.is_gpu_available:
        #     exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        keras.backend.clear_session()
        if self.use_pre_trained_model:
            self.logger.warning("Use pre-trained ncpp")
            self.model = keras.models.load_model(self.pre_trained_model_path, custom_objects={"MultiHeadAtten": MultiHeadAtten})
            if self.freeze_train_model:
                for layer in self.model.layers:
                    if layer.name.startswith("fix"):
                        layer.trainable = False
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.opm_init_lr)
            self.model.compile(optimizer=optimizer, loss="mse", metrics=self.metrics)

        else:
            self.model, lr_schedule = self.build_model()
            if (self.verbose == True):
                pass
                # self.ncpp.summary()
        # self.ncpp.save_weights(self.output_path + '/fcn/model_init.hdf5')
        # self.logger.info(
        #     f"train_data has \033[1;34;34m{x_train.shape[0]}\033[0m rows and \033[1;34;34m{x_train.shape[1]}\033[0m columns.")
        mini_batch_size = int(min(self.cpu_char_x_train.shape[0] / 10, self.batch_size))


        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.2, epsilon=0.0001, cooldown=10, min_lr=1e-6, verbose=0)


        model_checkpoint = ModelCheckpoint(
            filepath='%s/fcn-vec/%s_{epoch:d}_{val_loss:.3f}_checkpoint.hdf5' % (self.output_path,"checkpoints"), verbose=0,
            monitor='val_loss', save_best_only=True, save_weights_only=False, period=1000)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.output_path + "/logs/", histogram_freq=1)
        callback_list = [model_checkpoint, tensorboard, LogPrintCallback(interval=100)]

        input_x_data = {"mem_numer_input": self.mem_numer_x_train, "cpu_numer_input": self.cpu_numer_x_train,
                              "system_char_input": self.system_char_x_train,
                              "system_numer_input": self.system_numer_x_train,
                              "workload_char_input": self.workload_char_x_train}

        input_x_test = {"mem_numer_input": self.mem_numer_x_test, "cpu_numer_input": self.cpu_numer_x_test,
                                "system_char_input": self.system_char_x_test,
                                "system_numer_input": self.system_numer_x_test,
                                "workload_char_input": self.workload_char_x_test}
        if not self.use_pre_train_attmodel:


            hist = self.model.fit(input_x_data, self.y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(input_x_test, self.y_test), callbacks=callback_list, workers=16)
            # self.attention_part_model( "char" , sample_num=self.sample_index)
            # self.attention_part_model("mem_num", sample_num=self.sample_index)
            # self.attention_part_model("cross_group", sample_num=self.sample_index)
            # self.attention_part_model("cpu_num", sample_num=self.sample_index)
            # self.attention_part_model("system_num", sample_num=self.sample_index)
        else:
            # self.attention_part_model( "char" , sample_num=self.sample_index)
            # self.attention_part_model("mem_num", sample_num=self.sample_index)
            # self.attention_part_model("cross_group", sample_num=self.sample_index)
            # self.attention_part_model("cpu_num", sample_num=self.sample_index)
            # self.attention_part_model("system_num", sample_num=self.sample_index)
            sys.exit(0)

        log = pd.DataFrame(hist.history)

        self.logger.info("Finished training the ncpp")
        # self.save_coef()
        if self.is_save_model:
            if self.train_with_all_data:
                save_path = os.path.join(self.output_path, "ncpp").replace("\\", "/")
            else:
                save_path = os.path.join(self.kth_fold_save_path, "ncpp").replace("\\", "/")
            log.to_csv(save_path + "/train_hist.csv")
            plot_epochs_metric(hist, save_path + "/loss.png")
            plot_epochs_metric(hist, save_path + "/mae.png", metric="mean_absolute_error")
            plot_epochs_metric(hist, save_path + "/mse.png", metric="mean_squared_error")
            plot_epochs_metric(hist, save_path + "/mape.png", metric="my mpe")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.join(save_path, f"{self.select_model}.hdf5").replace("\\", "/")
            x_train_save_name = os.path.join(save_path, "processed_x_train.csv").replace("\\", "/")
            # x_train = pd.concat([char_x_train, numer_x_train], axis=1)
            # x_train.to_csv(x_train_save_name, index=False)
            with open(os.path.join(self.config_save_path, "config.yaml").replace("\\", "/"), 'w') as f:
                yaml.dump(self.configs, f)
            self.model.save(save_name)
            self.logger.warning(f"saving ncpp to: {save_path}")
        else:
            self.logger.info("train without saving ncpp")

        return self.model, log

    def train_without_validate(self):
        # if not tf.test.is_gpu_available:
        #     exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training


        self.logger.info("")


        keras.backend.clear_session()
        if self.use_pre_trained_model:
            self.logger.warning("Use pre-trained ncpp")
            self.model = keras.models.load_model(self.pre_trained_model_path, custom_objects={"MultiHeadAtten": MultiHeadAtten})
            if self.freeze_train_model:
                for layer in self.model.layers:
                    if layer.name.startswith("fix"):
                        layer.trainable = False
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.opm_init_lr)
            self.model.compile(optimizer=optimizer, loss="mse", metrics=self.metrics)

        else:
            self.model, lr_schedule = self.build_model()
            if (self.verbose == True):
                pass
                # self.ncpp.summary()
        # self.ncpp.save_weights(self.output_path + '/fcn/model_init.hdf5')
        # self.logger.info(
        #     f"train_data has \033[1;34;34m{x_train.shape[0]}\033[0m rows and \033[1;34;34m{x_train.shape[1]}\033[0m columns.")
        mini_batch_size = int(min(self.cpu_char_x_train.shape[0] / 10, self.batch_size))


        reduce_lr = ReduceLROnPlateau(
            monitor='mean_squared_error', patience=self.patience, mode='auto', factor=self.factor, cooldown=self.cooldown, min_lr=self.min_lr, verbose=0)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.opm_init_lr, decay_steps=self.decay_steps, decay_rate=self.decay_rate)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.opm_init_lr,
                                                                     decay_steps=self.decay_steps,
                                                                     decay_rate=self.decay_rate)


        checkpoint_dir = os.path.join(self.output_path, "weights")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        filepath = os.path.join(checkpoint_dir, "{epoch:d}_{mean_squared_error:4f}_checkpoint.hdf5").replace("\\", "/")

        model_checkpoint = ModelCheckpoint(filepath=filepath, verbose=0,
                                           monitor='mean_squared_error', save_best_only=True,
                                           save_weights_only=False, period=100)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.output_path + "/logs/", histogram_freq=1)
        callback_list = [model_checkpoint, tensorboard, LogPrintCallback(interval=100)]



        # custom_data_generator = CustomDataGenerator((char_x_train, numer_x_train), y_train, mini_batch_size)

        # hist = self.fcn.fit_generator(custom_data_generator, epochs=self.nb_epochs,
        #                                 verbose=self.verbose, validation_data=((char_x_test,  numer_x_test), self.y_test),
        #                                 callbacks=callback_list)
        strategy = tf.distribute.MirroredStrategy()
        # input_x_data = (self.mem_char_x_train, self.mem_numer_x_train, self.cpu_char_x_train, self.cpu_numer_x_train,
        #                 self.system_char_x_train, self.system_numer_x_train, self.workload_char_x_train, self.workload_numer_x_train)
        input_x_data = {"mem_numer_input": self.mem_numer_x_train, "cpu_numer_input": self.cpu_numer_x_train,
                              "system_char_input": self.system_char_x_train,
                              "system_numer_input": self.system_numer_x_train,
                              "workload_char_input": self.workload_char_x_train}


        if not self.use_pre_train_attmodel:


            hist = self.model.fit(input_x_data, self.y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=0, callbacks=callback_list)
        # self.plot_epochs_metric(hist, self.output_path + "/loss.png")
        # collected_all_values = {"mem_num": attention_weight_collector_mem.attention_weights,
        #                         "cpu_num": attention_weight_collector_cpu.attention_weights,
        #                         "system_num": attention_weight_collector_system.attention_weights,
        #                         "char": attention_weight_collector_char.attention_weights,
        #                         "cross_group": attention_weight_collector_cross_group.attention_weights}
        #     self.attention_part_model("char", sample_num=self.sample_index)
        #     self.attention_part_model("mem_num", sample_num=self.sample_index)
        #     self.attention_part_model("cross_group", sample_num=self.sample_index)
        #     self.attention_part_model("cpu_num", sample_num=self.sample_index)
        #     self.attention_part_model("system_num", sample_num=self.sample_index)
        else:
            self.logger.info("visualize attention")

            # self.attention_part_model("char", sample_num=self.sample_index)
            # self.attention_part_model("mem_num", sample_num=self.sample_index)
            # self.attention_part_model("cross_group", sample_num=self.sample_index)
            # self.attention_part_model("cpu_num", sample_num=self.sample_index)
            # self.attention_part_model("system_num", sample_num=self.sample_index)
            sys.exit(0)

        log = pd.DataFrame(hist.history)
        log.to_csv(self.output_path + "/trainlog")
        self.logger.info("Finished training the ncpp")
        # self.save_coef()
        if self.is_save_model:
            if self.train_with_all_data:
                save_path = os.path.join(self.output_path, "ncpp").replace("\\", "/")
            else:
                save_path = os.path.join(self.kth_fold_save_path, "ncpp").replace("\\", "/")
            plot_epochs_metric(hist, save_path + "/loss.png")
            log.to_csv(save_path + "/trainloss.csv")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.join(save_path, f"{self.select_model}.hdf5").replace("\\", "/")
            x_train_save_name = os.path.join(save_path, "processed_x_train.csv").replace("\\", "/")
            # x_train = pd.concat([char_x_train, numer_x_train], axis=1)
            # x_train.to_csv(x_train_save_name, index=False)
            with open(os.path.join(self.config_save_path, "config.yaml").replace("\\", "/"), 'w') as f:
                yaml.dump(self.configs, f)
            self.model.save(save_name)
            self.logger.warning(f"saving ncpp to: {save_path}")
        else:
            self.logger.info("train without saving ncpp")

        return self.model, log

    def attention_part_model(self, output_layer_name, feature_names_list=None, sample_num=0):
        test_x_data = {"mem_numer_input": self.mem_numer_x_train, "cpu_numer_input": self.cpu_numer_x_train,
                              "system_char_input": self.system_char_x_train,
                              "system_numer_input": self.system_numer_x_train,
                              "workload_char_input": self.workload_char_x_train}
        if self.use_pre_train_attmodel:
            model_path = self.attention_model_path
            model = keras.models.load_model(model_path, custom_objects={"MultiHeadAtten": MultiHeadAtten})
        else:
            model = self.model


        if output_layer_name == "char":
            if self.pooled_char_feature:
                feature_names_list = [self.report_feature_mapping.get(item[0], item[0]) for sublist in self.char_token_order.values() for item in sublist]
            else:
                feature_names_list = genereate_feature_list(self.char_token_order, self.report_feature_mapping)
                feature_names_list = list(self.report_feature_mapping.get(item[0], item[0]) for item in feature_names_list)
        elif output_layer_name == "mem_num":
            feature_names_list = list(self.report_feature_mapping.get(item[0], item[0]) for item in self.numeric_token_order["Memory_info"])
        elif output_layer_name == "cpu_num":
            feature_names_list = list(self.report_feature_mapping.get(item[0], item[0]) for item in self.numeric_token_order["Processor_info"])
        elif output_layer_name == "system_num":
            feature_names_list = list(self.report_feature_mapping.get(item[0], item[0]) for item in self.numeric_token_order["System_info"])
        elif output_layer_name == "cross_group":
            feature_names_list = ["mem_num", "cpu_num", "system_num", "char"]


        attention_model = Model(inputs=model.input, outputs=model.get_layer(output_layer_name).output)
        _, attention_model_output, or_QKo, Qi, x, w = attention_model.predict(test_x_data, verbose=0)
        if self.train_with_all_data:
            output_path = self.output_path
        else:
            output_path = self.kth_fold_save_path
        visualized_attention(or_QKo,attention_model_output, Qi, x, w,output_path, name=f"{output_layer_name}_attention.png", feature_name_list=feature_names_list, sample_num=sample_num)

    def predict(self):
        model_save_path = os.path.join(self.final_path, "ncpp").replace("\\", "/")
        save_name = os.path.join(model_save_path, f"{self.select_model}.hdf5").replace("\\", "/")
        # register customed_mse to custom_object_scope
        with custom_object_scope({'customed_mse': self.customed_mse}):
            model = keras.models.load_model(save_name, custom_objects={"MultiHeadAtten": MultiHeadAtten})
        # input_x_data = (self.mem_char_x_test, self.mem_numer_x_test, self.cpu_char_x_test, self.cpu_numer_x_test,
        #                 self.system_char_x_test, self.system_numer_x_test, self.workload_char_x_test, self.workload_numer_x_test)
        input_x_data = {"mem_numer_input": self.mem_numer_x_test, "cpu_numer_input": self.cpu_numer_x_test,
                                "system_char_input": self.system_char_x_test,
                                "system_numer_input": self.system_numer_x_test,
                                "workload_char_input": self.workload_char_x_test}
        y_pred = model.predict(input_x_data, verbose=0)
        # y_pred = self.ncpp.predict((self.char_x_test, self.numer_x_test))
        # convert the predicted from binary to integer
        #y_pred = np.argmax(y_pred, axis=1)
        # keras.backend.clear_session()
        return y_pred


    @calculate_running_time
    def run(self, train_with_all_data=False, result=None):
        self.logger.debug("Begin training the ncpp")
        if not train_with_all_data:
            self.mem_char_x_train = self.get_train_data(self.mem_processed_char_feature)
            self.mem_numer_x_train = self.get_train_data(self.mem_processed_numa_features)
            self.cpu_char_x_train = self.get_train_data(self.cpu_processed_char_feature)
            self.cpu_numer_x_train = self.get_train_data(self.cpu_processed_numa_features)
            self.system_char_x_train = self.get_train_data(self.system_processed_char_feature)
            self.system_numer_x_train = self.get_train_data(self.system_processed_numa_features)
            self.workload_char_x_train = self.get_train_data(self.workload_processed_char_feature)
            self.workload_numer_x_train = self.get_train_data(self.workload_processed_numa_features)

            self.mem_char_x_test = self.get_test_data(self.mem_processed_char_feature)
            self.mem_numer_x_test = self.get_test_data(self.mem_processed_numa_features)
            self.cpu_char_x_test = self.get_test_data(self.cpu_processed_char_feature)
            self.cpu_numer_x_test = self.get_test_data(self.cpu_processed_numa_features)
            self.system_char_x_test = self.get_test_data(self.system_processed_char_feature)
            self.system_numer_x_test = self.get_test_data(self.system_processed_numa_features)
            self.workload_char_x_test = self.get_test_data(self.workload_processed_char_feature)
            self.workload_numer_x_test = self.get_test_data(self.workload_processed_numa_features)

            self.y_train = self.get_train_data(self.processed_labels)
            self.y_test = self.get_test_data(self.processed_labels)

            self.mem_char_x_train_shape = self.mem_char_x_train.shape[1:]
            self.cpu_char_x_train_shape = self.cpu_char_x_train.shape[1:]
            self.system_char_x_train_shape = self.system_char_x_train.shape[1:]
            self.workload_char_x_train_shape = self.workload_char_x_train.shape[1:]
            self.mem_numer_x_train_shape = self.mem_numer_x_train.shape[1:]
            self.cpu_numer_x_train_shape = self.cpu_numer_x_train.shape[1:]
            self.system_numer_x_train_shape = self.system_numer_x_train.shape[1:]
            self.workload_numer_x_train_shape = self.workload_numer_x_train.shape[1:]



            model, hist = self.train()
            self.y_predict = self.predict()
            result = self.evaluate(self.y_test, self.y_predict)
            if self.isplot:
                save_path = self.output_path
                Train_predict_compare(self.configs, self.y_predict, self.y_test, save_path)
        else:
            self.logger.warning("Train ncpp with all the data, without validation!")
            self.mem_char_x_train = self.data_reshape(self.mem_processed_char_feature)
            self.mem_numer_x_train = self.data_reshape(self.mem_processed_numa_features)
            self.cpu_char_x_train = self.data_reshape(self.cpu_processed_char_feature)
            self.cpu_numer_x_train = self.data_reshape(self.cpu_processed_numa_features)
            self.system_char_x_train = self.data_reshape(self.system_processed_char_feature)
            self.system_numer_x_train = self.data_reshape(self.system_processed_numa_features)
            self.workload_char_x_train = self.data_reshape(self.workload_processed_char_feature)
            self.workload_numer_x_train = self.data_reshape(self.workload_processed_numa_features)
            self.y_train = self.processed_labels

            self.mem_char_x_train_shape = self.mem_char_x_train.shape[1:]
            self.cpu_char_x_train_shape = self.cpu_char_x_train.shape[1:]
            self.system_char_x_train_shape = self.system_char_x_train.shape[1:]
            self.workload_char_x_train_shape = self.workload_char_x_train.shape[1:]
            self.mem_numer_x_train_shape = self.mem_numer_x_train.shape[1:]
            self.cpu_numer_x_train_shape = self.cpu_numer_x_train.shape[1:]
            self.system_numer_x_train_shape = self.system_numer_x_train.shape[1:]
            self.workload_numer_x_train_shape = self.workload_numer_x_train.shape[1:]


            model, hist= self.train_without_validate()
            self.copy_model()
            sys.exit(0)

        return result, hist

    def get_train_data(self, data):
        if data.empty:
            return pd.DataFrame()
        data = data.loc[self.train_indices]
        data.reset_index(drop=True)
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        return data_reshape

    def data_reshape(self, data):
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        return data_reshape

    def get_test_data(self, data):
        if data.empty:
            return pd.DataFrame()
        data = data.loc[self.test_indices]
        data.reset_index(drop=True)
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








