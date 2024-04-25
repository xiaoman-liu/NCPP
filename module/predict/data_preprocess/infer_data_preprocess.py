#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/22/2023 9:48 AM
# @Author  : xiaomanl
# @File    : string2vec.py
# @Software: PyCharm

from collections import OrderedDict
import yaml

from src.inference.utils import calculate_running_time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.inference.utils import calculate_running_time, mkdir
from src.inference.utils.model_utils import TextTokenizer
from sklearn.compose import ColumnTransformer
from pathlib import Path
import joblib
import  glob





class FeatureEncoder():
    def __init__(self, configs):
        self.configs = configs
        self.logger = logging.getLogger("FeatureEmbedding")
        self.processed_numa_features = pd.DataFrame()
        self.processed_char_feature = pd.DataFrame()
        self.processed_features = pd.DataFrame()
        feature_config = self.configs["select_model"] + "_feature_config"
        self.feature_rule_config = self.configs[feature_config]

        self.max_char_length = 0
        self.model_configs = configs["data_convert_config"] if "data_convert_config" in configs.keys() else None
        self.output_path = configs["output_path"]
        self.feature_order = configs["feature_order"]
        self.feature_encoder_save_path = os.path.join(self.output_path, "model", "encoder").replace("\\", "/")
        self.select_model = self.configs["select_model"]
        self.encoder_path = configs["encoder_path"]
        self.no_embedding_model_list = self.configs["no_embedding_model_list"]


    def numpy_dtype_representer(self, dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.items()))

    def dict_representer(self,dumper, data):
        return dumper.represent_dict(data.items())

    def scale_min_max(self, feature):
        """
        Scale the continuous features to the range of [-1, 1].
        """
        feature = feature.astype(float)
        col = feature.columns.to_list()[0]
        scaler = joblib.load(os.path.join(self.encoder_path, f"{col}#minmax.pkl").replace("\\", "/"))
        scaled_features = scaler.normalize(feature)

        # feature = scaled_features.loc[:, np.isfinite(scaled_features).all()]
        # self.logger.info("Finished data normalization using Min_max_scaler method.")
        return scaled_features

    def scale_normalize(self, feature):
        """
        Scale the value to have zero mean and unit variance
        :return: Scaled continuous features
        """
        try:
            feature = feature.astype(float)
            col = feature.columns.to_list()[0]
            file = glob.glob(self.encoder_path + f'/{col}nor*.pkl')[0]
            scaler = joblib.load(file)
            result = scaler.normalize(feature)

            selected_feature = result
        except Exception as e:
            selected_feature = pd.DataFrame(0, index=feature.index, columns=feature.columns)
            self.logger.warning(f"Error in Normalization: {e}")
        return selected_feature



    def without_normalization(self, feature):
        self.logger.warning(f"You don't have use any scaler methods for {feature.columns.to_list()[0]}")
        return feature


    def Tokenizer(self, feature):
        col = feature.columns.to_list()[0]
        tokenizer = joblib.load(os.path.join(self.encoder_path, f"{col}#tokenizer.pkl").replace("\\", "/"))
        # self.logger.info("Use Tokenizer to process the string feature.")
        feature = tokenizer.transform(feature)

        return feature

    def data_transformation(self):
        if self.model_configs["use_ploy"]:
            self.logger.info("Use the ployniminal transformation.")
            ploy_degree = self.model_configs["ploy_config"]["ploy_degree"]
            poly = PolynomialFeatures(degree=ploy_degree)
            poly_features = poly.fit_transform(self.processed_features)
            col_names = ['x{}'.format(i) for i in range(poly_features.shape[1])]
            self.processed_features = pd.DataFrame(poly_features, columns=col_names)

        if self.model_configs["use_lnx"]:
            self.logger.info("Use log transformation.")
            epsilon = 1e-6  # Define a small constant value
            self.processed_features = pd.DataFrame(np.log(self.processed_features.values + epsilon + 1),
                                                    columns=self.processed_features.columns)
            #self.continuous_features = pd.DataFrame(np.sqrt(self.continuous_features.values),columns=self.continuous_features.columns)
            # from scipy.stats import boxcox
            #
            # self.continuous_features = pd.DataFrame(boxcox(self.continuous_features.values)[0],
            #                                         columns=self.continuous_features.columns)
            # from sklearn.preprocessing import QuantileTransformer
            #
            # quantile_transformer = QuantileTransformer(output_distribution='normal')
            # self.continuous_features = pd.DataFrame(quantile_transformer.fit_transform(self.continuous_features.values),
            #                                         columns=self.continuous_features.columns)
        if self.model_configs["use_kpca"]:
            self.logger.info("Use Kernel PCA transformation.")
            n_components = self.model_configs["kernelPCA_config"]["n_components"]
            kernel = self.model_configs["kernelPCA_config"]["kernel"]
            gamma = self.model_configs["kernelPCA_config"]["gamma"]
            kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
            self.continuous_features = kpca.fit_transform(self.continuous_features)
            self.continuous_features = pd.DataFrame(self.continuous_features)

    def Onehot_encode(self, feature):
        col = feature.columns.to_list()[0]
        encoder = joblib.load(os.path.join(self.encoder_path, f"{col}#onehot.pkl").replace("\\", "/"))

        processed_feature= encoder.transform(feature)

        return processed_feature

    def Label_encode(self, feature):

        col = feature.columns.to_list()[0]
        if self.select_model == "XgboostModel" or self.select_model == "RandomForest":
            encoder = joblib.load(os.path.join(self.encoder_path, f"{col}label.pkl").replace("\\", "/"))
        else:
            encoder = joblib.load(os.path.join(self.encoder_path, f"{col}#label.pkl").replace("\\", "/"))
        processed_feature= encoder.transform(feature)
        processed_feature = processed_feature.astype(float)

        return processed_feature

    def align_features(self, features):
        """
        Align the features by the column names.
        """
        useful_cols = [key for key, value in self.feature_rule_config.items() if value.get("used_in_training") == 1]
        current_columns = features.columns.to_list()

        # Assuming self.filtered_features is a pandas DataFrame and self.column_types is a dict mapping column names to their types
        missing_columns = []
        for col in useful_cols :
            if col not in current_columns and col != "RESULT.TestName":
                missing_columns.append(col)
                # Add the column with default values based on its supposed type
                if self.feature_rule_config[col]["data_type"] == 'string':
                    features[col] = "null"
                else:
                    features[col] = 0
        self.logger.warning(f"Missing columns: {missing_columns} in the input data. Add them with default values.")

        return features

    @calculate_running_time
    def run(self, filtered_features):
        """
        Run the data preprocessor.
        """
        process_methods = {
            "Min_max_scaler": self.scale_min_max,
            "Normalization_scaler": self.scale_normalize,
            "Tokenizer": self.Tokenizer,
            "Onehot_encoding": self.Onehot_encode,
            "Label_encoding": self.Label_encode

        }




        yaml.add_representer(OrderedDict, self.dict_representer)
        yaml.add_representer(np.dtype, self.numpy_dtype_representer)
        self.logger.info("before align the features, the shape of the filtered features is {}".format(filtered_features.shape))

        self.filtered_features = self.align_features(filtered_features)
        self.logger.info("after align the features, the shape of the filtered features is {}".format(self.filtered_features.shape))
        # Choose scaler method
        if self.select_model in self.no_embedding_model_list:
            for col in self.feature_order:
                scaler_method = self.feature_rule_config[col]["processing_method"]["name"]
                self.logger.info(f"parse {col} with {scaler_method}")
                single_feature = process_methods.get(scaler_method, self.scale_normalize)(self.filtered_features[[col]])
                if not single_feature.empty:
                    self.processed_features = pd.concat([self.processed_features, single_feature], axis=1)
                # scaler_methods.get(self.scaler_method, self.without_normalization)()
            if self.model_configs:
                self.data_transformation()
            self.processed_features.to_csv()
            return self.processed_features

        else:
            for col in self.feature_order:
                scaler_method = self.feature_rule_config[col]["processing_method"]["name"]
                self.logger.info(f"parse {col} with {scaler_method}")
                single_feature = process_methods.get(scaler_method, self.scale_normalize)(self.filtered_features[[col]])
                if not single_feature.empty:
                    if scaler_method == "Min_max_scaler" or scaler_method == "Normalization_scaler":
                        self.processed_numa_features = pd.concat([self.processed_numa_features, single_feature], axis=1)
                    else:
                        self.processed_char_feature = pd.concat([self.processed_char_feature, single_feature], axis=1)
            column_names = self.processed_char_feature.columns.tolist()
            new_column_names = [f'col_{i}' for i in range(len(column_names))]
            self.processed_char_feature.columns = new_column_names
            self.logger.info("The shape of the processed char feature is {}".format(self.processed_char_feature.shape))
            self.logger.info("The shape of the processed continuous feature is {}".format(self.processed_numa_features.shape))
            return (self.processed_numa_features, self.processed_char_feature)











