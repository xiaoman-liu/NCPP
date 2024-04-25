#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/22/2023 9:48 AM
# @Author  : xiaomanl
# @File    : string2vec.py
# @Software: PyCharm
import sys
from collections import OrderedDict
import yaml

from src.train.train_utils import calculate_running_time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.inference.utils import calculate_running_time, read_class_config, mkdir
from src.inference.utils.model_utils import MinMaxScaler, NorScaler, OneHotEncoder, TextTokenizer
from sklearn.compose import ColumnTransformer
from pathlib import Path
import joblib
import glob





class GroupFeatureEmbedding():
    def __init__(self, configs, config_file="./config/group.yaml"):
        self.configs = configs
        self.config_file = config_file
        self.configs.update(read_class_config(Path(__file__).resolve().parent, self.config_file ))
        self.logger = logging.getLogger("GroupFeatureEmbedding")
        self.mem_processed_numa_features = pd.DataFrame()
        self.mem_processed_char_feature = pd.DataFrame()
        self.cpu_processed_numa_features = pd.DataFrame()
        self.cpu_processed_char_feature = pd.DataFrame()
        self.system_processed_numa_features = pd.DataFrame()
        self.system_processed_char_feature = pd.DataFrame()
        self.workload_processed_numa_features = pd.DataFrame()
        self.workload_processed_char_feature = pd.DataFrame()
        self.numa_processed_features = pd.DataFrame()
        self.char_processed_features = pd.DataFrame()
        self.group_config = self.configs["feature_group_config"]

        feature_config = self.configs["select_model"] + "_feature_config"
        self.feature_rule_config = self.configs[feature_config]

        self.max_char_length = 0
        self.config_save_path = self.configs["config_save_path"]
        self.output_path = configs["output_path"]
        self.feature_encoder_save_path = os.path.join(self.output_path, "model", "encoder").replace("\\", "/")
        mkdir(self.feature_encoder_save_path)
        self.select_model = self.configs["select_model"]
        self.encoder_path = configs["encoder_path"]
        self.no_embedding_model_list = self.configs["no_embedding_model_list"]
        self.feature_order = configs["feature_order"]
        self.if_label_scale = self.configs["if_label_scale"]

        # self.label_scaler = MinMaxScaler()
        self.workload_name = self.configs["workload_names"][0]
        self.label_scale = self.configs["label_scale"].get(self.workload_name, 1)


    def numpy_dtype_representer(self, dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.items()))

    def dict_representer(self,dumper, data):
        return dumper.represent_dict(data.items())

    def feature_reduction(self, feature):
        if self.configs["drop_for_linear"]:
            drop_columns = ["kubernetes.pod_id", "RESULT.kubernetes.host", "RESULT.cluster-name", "SVR.CPU.CPU Model", "SVR.CPU.Microarchitecture", "SVR.CPU.Prefetchers", "SVR.CPU.Turbo", "SVR.Accelerators", "SVR.Accelerators", "SVR.Power.Frequency Governer", "SVR.Power.Power & Perf Policy", "Measure.DIMM.Population","Measure.DIMM.PartNo"
            "RESULT.WorkloadPreset", "SVR.CPU.NUMA Node(s)", "SVR.ISA", "META.metadata.cscope.stepping", "RESULT.IterationIndex"]
        else:
            drop_columns=[]
        intersect_colums = list(set(feature.columns) & set(drop_columns))
        # intersect_colums = list(feature.columns)
        reducted_feature = feature.drop(columns=intersect_colums)
        # reducted_feature = feature
        return reducted_feature
    def scale_min_max(self, feature):
        """
        Scale the continuous features to the range of [-1, 1].
        """
        feature = feature.astype(float)
        col = feature.columns.to_list()[0]
        scaler = joblib.load(os.path.join(self.encoder_path, f"{col}#minmax.pkl").replace("\\", "/"))
        print(f"{col} use minmax, min is {scaler.min_value}, max is {scaler.max_value}")
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
            print(f"{col} use normalization, mean is {scaler.mean_value}, std is {scaler.std_value}")
            result = scaler.normalize(feature)

            selected_feature = result
        except Exception as e:
            selected_feature = pd.DataFrame(0, index=feature.index, columns=feature.columns)
            self.logger.warning(f"Error in Normalization: {e}")
        return selected_feature

    def save_feature_rules_to_yaml(self, features, method, feature_mapping=None, feature_columns = None):
        # Save information in YAML format

        info = OrderedDict()
        if features.empty:
            self.logger.warning(f"after {features.columns}, {features.columns} got none values.")
            return
        col = feature_columns if method == "Onehot_encoding" or method == "Tokenizer" else features.columns.to_list()[0]
        info[col] = OrderedDict()
        info[col]["used_in_training"] = self.feature_rule_config[col]["used_in_training"]
        info[col]["processing_method"] = {}
        info[col]["data_type"] = self.feature_rule_config[col]["data_type"]
        info[col]["processing_method"]["name"] = method
        if method == "Min_max_scaler":
            params = {'mean_value': features[col].mean(),
                      'min_value': features[col].min(),
                      'max_value': features[col].max()}
        elif method == "Normalization_scaler":
            params = {'mean_value': float(features[col].mean()),
                      'std_value': features[col].std()}
        else:
            params = {"mapping_file": f"{method}_mapping.yaml",
                      "mapping_key": col}

        info[col]["processing_method"]["param"] = params
        result = info

        # Save information to YAML file
        output_path = self.configs["output_path"]
        self.configs["feature_config_save_path"] = os.path.join(self.config_save_path, "features_processor_config.yaml").replace("\\", "/")
        if not os.path.exists(self.configs["feature_config_save_path"]):
            os.makedirs(os.path.dirname(self.configs["feature_config_save_path"]), exist_ok=True)
            with open(self.configs["feature_config_save_path"], 'w') as f:
                yaml.dump({"features_processor_config":OrderedDict()}, f)

        with open(self.configs["feature_config_save_path"], 'r+') as file:
            data = yaml.load(file, Loader=yaml.Loader)
            merged_dict = OrderedDict()
            merged_dict["features_processor_config"] = OrderedDict()
            merged_dict["features_processor_config"].update(data["features_processor_config"])
            merged_dict["features_processor_config"].update(result)
            # result = {**data, **result}
            file.seek(0)
            yaml.dump(merged_dict, file, default_flow_style=False, indent=4)
            file.truncate()
        # self.logger.info(f'save feature config to {self.configs["feature_config_save_path"]}')

        if feature_mapping is not None:
            mapping_name_dict = {"Label_encoding":"LE_mapping", "Onehot_encoding":"OHE_mapping", "Tokenizer":"Tokenizer_mapping"}
            mapping_name = mapping_name_dict[method]
            self.configs["mappping_save_path"] = os.path.join(self.config_save_path,
                                                                    f"{method}_mapping.yaml").replace("\\", "/")
            if not os.path.exists(self.configs["mappping_save_path"]):
                os.makedirs(os.path.dirname(self.configs["mappping_save_path"]), exist_ok=True)

                with open(self.configs["mappping_save_path"], 'w') as f:
                    yaml.dump({mapping_name:OrderedDict()}, f)

            with open(self.configs["mappping_save_path"], 'r+') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                merged_dict = OrderedDict()
                merged_dict[mapping_name] = OrderedDict()
                merged_dict[mapping_name].update(data[mapping_name])
                merged_dict[mapping_name].update(feature_mapping)
                file.seek(0)
                yaml.dump(merged_dict, file, default_flow_style=False, indent=4)
                file.truncate()
            # self.logger.info(f'save feature config to {self.configs["mappping_save_path"]}')



    def without_normalization(self, feature):
        self.logger.warning(f"You don't have use any scaler methods for {feature.columns.to_list()[0]}")
        return feature


    def Tokenizer(self, feature):
        col = feature.columns.to_list()[0]
        feature = feature.astype(str)
        tokenizer = TextTokenizer(feature)
        print(f"tokenizer for {col} is {tokenizer}")
        processed_feature = tokenizer.fit_transform(feature)

        # joblib.dump(tokenizer, os.path.join(self.feature_encoder_save_path,f"{col}#tokenizer.pkl").replace("\\", "/"))
        self.save_feature_rules_to_yaml(processed_feature, "Tokenizer", feature_columns=col)
        # self.logger.info("Finished data Tokenizer.")
        return processed_feature

    def Onehot_encode(self, feature):
        col = feature.columns.to_list()[0]
        encoder = joblib.load(os.path.join(self.encoder_path, f"{col}#onehot.pkl").replace("\\", "/"))

        processed_feature= encoder.transform(feature)

        return processed_feature

    def Label_encode(self, feature):

        col = feature.columns.to_list()[0]
        encoder = joblib.load(os.path.join(self.encoder_path, f"{col}#label.pkl").replace("\\", "/"))
        processed_feature = encoder.transform(feature)
        processed_feature = processed_feature.astype(float)

        return processed_feature



    @calculate_running_time
    def run(self, filtered_features, labels=None):
        """
        Run the data preprocessor.
        """
        process_methods = {
            "Min_max_scaler": self.scale_min_max,
            "Normalization_scaler": self.scale_normalize,
            "Tokenizer": self.Tokenizer,
            "Onehot_encoding": self.Onehot_encode,
        }

        # self.filtered_features = self.feature_reduction(filtered_features)
        self.filtered_features = filtered_features

        yaml.add_representer(OrderedDict, self.dict_representer)
        yaml.add_representer(np.dtype, self.numpy_dtype_representer)


        self.logger.info(
            f"The shape of processed feature data is: \033[1;34;34m{self.filtered_features.shape[0]}\033[0m rows and \033[1;34;34m{self.filtered_features.shape[1]}\033[0m columns.")
        # Choose scaler method

        # Choose scaler method

        for col in self.feature_order:
            if col in ['RESULT.WorkloadName', 'RESULT.TestName', ""]:
                self.feature_rule_config[col]["used_in_training"] = 0
            if col in self.feature_rule_config and self.feature_rule_config[col]["used_in_training"] == 1:
                scaler_method = self.feature_rule_config[col]["processing_method"]["name"]
                self.logger.info(f"parse {col} with {scaler_method}")

                single_feature = process_methods.get(scaler_method, self.scale_normalize)(self.filtered_features[[col]])
                if not single_feature.empty:
                    if scaler_method == "Min_max_scaler" or scaler_method == "Normalization_scaler":
                        if col in self.group_config["Memory_info"]:
                            self.mem_processed_numa_features = pd.concat([self.mem_processed_numa_features, single_feature], axis=1)
                        elif col in self.group_config["Processor_info"]:
                            self.cpu_processed_numa_features = pd.concat([self.cpu_processed_numa_features, single_feature], axis=1)
                        elif col in self.group_config["System_info"]:
                            self.system_processed_numa_features = pd.concat([self.system_processed_numa_features, single_feature], axis=1)
                        elif col in self.group_config["Workload_info"]:
                            self.workload_processed_numa_features = pd.concat([self.workload_processed_numa_features, single_feature], axis=1)
                        self.numa_processed_features = pd.concat([self.numa_processed_features, single_feature], axis=1)
                    else:
                        if col in self.group_config["Memory_info"]:
                            self.mem_processed_char_feature = pd.concat([self.mem_processed_char_feature, single_feature], axis=1)
                        elif col in self.group_config["Processor_info"]:
                            self.cpu_processed_char_feature = pd.concat([self.cpu_processed_char_feature, single_feature], axis=1)
                        elif col in self.group_config["System_info"]:
                            self.system_processed_char_feature = pd.concat([self.system_processed_char_feature, single_feature], axis=1)
                        elif col in self.group_config["Workload_info"]:
                            self.workload_processed_char_feature = pd.concat([self.workload_processed_char_feature, single_feature], axis=1)
                        self.char_processed_features = pd.concat([self.char_processed_features, single_feature], axis=1)

                    self.logger.info(f"processed feature shape is {single_feature.shape}")

            else:
                continue

        self.logger.info(f"processed mem feature shape is {self.mem_processed_numa_features.shape}")
        self.logger.info(f"processed cpu feature shape is {self.cpu_processed_numa_features.shape}")
        self.logger.info(f"processed system feature shape is {self.system_processed_numa_features.shape}")
        self.logger.info(f"processed workload feature shape is {self.workload_processed_numa_features.shape}")
        self.logger.info(f"processed numa feature shape is {self.numa_processed_features.shape}")
        self.logger.info(f"processed char feature shape is {self.char_processed_features.shape}")
        all_infer_process_feature = pd.concat([self.mem_processed_numa_features, self.mem_processed_char_feature, self.cpu_processed_numa_features, self.cpu_processed_char_feature,
                                               self.system_processed_numa_features, self.system_processed_char_feature, self.workload_processed_numa_features,
                                               self.workload_processed_char_feature, self.numa_processed_features, self.char_processed_features], axis=1)
        all_infer_process_feature.to_csv(os.path.join(self.output_path, "all_infer_process_feature.csv").replace("\\", "/"), index=False)
        self.logger.info(f"save all_infer_process_feature to {os.path.join(self.output_path, 'all_infer_process_feature.csv')}")
        # good_features = pd.read_csv(os.path.join(self.output_path, "good_all_infer_process_feature.csv").replace("\\", "/"))
        # columns = all_infer_process_feature.columns.to_list()
        # for col in columns:
        #     diff = all_infer_process_feature[col] != good_features[col]
        #     if diff.any():
        #         print(f"Differences in column {col}:")
        #         print(pd.concat([all_infer_process_feature[diff], good_features[diff]], keys=['df1', 'df2']))

        if self.if_label_scale:
            label_temp = labels.values

            label_scaler = joblib.load(os.path.join(self.encoder_path, 'labels_minmax.pkl').replace("\\", "/"))
            print("labels use minmax, min is {label_scaler.min_value}, max is {label_scaler.max_value}")
            label_temp = label_scaler.transform(label_temp)
            self.label = pd.DataFrame(label_temp, columns=labels.columns, index=labels.index)
        else:
            self.label = labels / self.label_scale

        return [[self.mem_processed_numa_features, self.mem_processed_char_feature],
                [self.cpu_processed_numa_features, self.cpu_processed_char_feature],
                [self.system_processed_numa_features, self.system_processed_char_feature],
                [self.workload_processed_numa_features, self.workload_processed_char_feature],
                [self.numa_processed_features, self.char_processed_features]], self.label




