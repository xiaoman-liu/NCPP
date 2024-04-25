#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/23/2022 9:28 AM
# @Author  : xiaomanl
# @File    : train_utils
# @Software: PyCharm

import yaml
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv
import time
from joblib import dump, load
from pathlib import Path


logger = logging.getLogger("UtilsModule")



def calculate_running_time(func):
    logger = logging.getLogger("ExecutionTime")
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        try:
            class_name = args[0].__class__.__name__
            logger.info(f"Time taken by {class_name}: {running_time:2f} seconds.")
        except:
            func_name = func.__name__
            logger.info(f"Time taken by {func_name}: {running_time:2f} seconds.")
        return result
    return wrapper

def generate_abs_path(current_dir, relative_path):
    abs_path = os.path.join(current_dir,relative_path).replace("\\", "/")
    return abs_path



def read_file(file_path):
    supported_formats = [".xlsx", ".csv"]
    file_format = os.path.splitext(file_path)[1]

    if file_format not in supported_formats:
        raise ValueError("File format not supported. Please use .xlsx or .csv")

    if file_format == ".xlsx":
        df = pd.read_excel(file_path, header=1)
        # df.dropna(axis=0, subset=['QDF/SSPEC'])
    elif file_format == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    else:
        raise ValueError("File format not supported. Please use .xlsx or .csv")

    return df

def generate_output_path(configs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if configs["infer_label"] and configs["infer_label"] != "test":
        output_path = os.path.join(configs["parent_output_path"],"milestones", configs["infer_label"],"/".join(configs["workload_names"]+ [configs["select_model"]] + [timestamp]))
    else:
        output_path = os.path.join(configs["parent_output_path"], "_".join(configs["workload_names"]), configs["select_model"], timestamp).replace("\\", "/")
    logging.info(f"Please checkout data in the path {output_path}")
    mkdir(output_path)

    return output_path

def read_config(root_dir, config_path, output_path):
    # Generate absolute paths
    config_path = generate_abs_path(root_dir, config_path)
    output_path = generate_abs_path(root_dir, output_path)

    # Read configuration files
    configs = read_yaml_files(config_path)

    # Read fcn configuration file
    model_config_path = generate_abs_path(root_dir, configs["model_path"] + "/config")
    encoder_path = generate_abs_path(root_dir, configs["model_path"] + "/encoder")


    model_config = read_yaml_files(model_config_path)
    logger.info(f"Read the fcn config from {model_config_path}")

    # Update configuration dictionary
    parent_output_path = output_path
    configs.update(model_config)
    configs.update({"parent_output_path": parent_output_path})
    configs.update({"output_path": generate_abs_path(root_dir, generate_output_path(configs))})
    configs.update({"encoder_path": encoder_path})
    configs.update({"config_save_path": os.path.join(configs["output_path"], "fcn", "config").replace("\\", "/")})
    root_dir = str(root_dir).replace("\\", "/")
    configs.update({"root_dir": root_dir})

    return configs

def read_yaml_files(dir_path):
    """
    Load YAML files and merge their data.
    :param dir_path: Directory path containing the YAML files.
    :return: Merged data from all the YAML files.
    """
    result = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".yaml"):
            file_path = os.path.join(dir_path, filename).replace("\\","/")
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    result.update(data)
            except Exception as e:
                logger.info(f"Error reading file {filename}: {e}")

    # Get all files in the current directory
    # files = os.listdir(dir_path)
    #
    # # Filter YAML files
    # yaml_files = [file for file in files if file.endswith('.yaml') or file.endswith('.yml')]
    #
    # # Read each YAML file
    # for yaml_file in yaml_files:
    #     file_path = os.path.join(dir_path, yaml_file)
    #     with open(file_path, 'r') as f:
    #         data = yaml.safe_load(f)
    #         result.update(data)
    return result

def read_yaml_file(path):
    """
    Load YAML files and merge their data.
    :param dir_path: Directory path containing the YAML files.
    :return: Merged data from all the YAML files.
    """
    result = {}
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                result.update(data)
        else:
            logger.error(f"File {path} does not exist.")
    except Exception as e:
                logger.info(f"Error reading file {path}: {e}")

    return result

def read_class_config(current_dir, config_file):

    data_filter_config_path = generate_abs_path(current_dir, config_file)
    class_config = read_yaml_file(data_filter_config_path)
    return class_config
def Train_predict_compare(configs, predict, label_test, save_path="output"):
    """
    compare with the true label with predict_orig label, calculate the accuracy
    :param predict:
    :param label_test:
    :return:
    """
    logging.info("draw a picture")
    plt.title("multiple linear fcn performance")
    plt.plot(np.arange(len(predict)), np.array(predict), 'ro-', label='predict_value')
    plt.plot(np.arange(len(label_test)), np.array(label_test), 'go-', label='true_value')
    plt.show()
    if configs["is_save_picture"]:
        plt.savefig(save_path)
    return

def log_assert(condition, message):
    if not condition:
        logger.error(f"AssertionError: {message}")
        assert condition, message
def mkdir(path):
    """

    :param path:
    :return:
    """
    if not os.path.exists(path.replace("\\", "/")):
        os.makedirs(path)
        logger.info("Directory '%s' created" % path)
    else:
        logger.info("Directory '%s' already exists" % path)
    return

def dict_to_csv(configs, data, type="", path=""):
    title = "/".join(configs['workload_names'])
    head = [title, "importance","percentage(%)"]
    file_path = os.path.join(path, "feature_ranking", "{}.csv".format(type)).replace("\\","/")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(head)
        total_importance = sum(data.values())
        for feature, importance in data.items():
            percent = importance / total_importance * 100
            writer.writerow([feature, importance, percent])

def generate_evaluate_metric(predict_label, true_label):
    # Calculate the Absolute Error (AE)
    ae = (predict_label["y_predict"].sub(true_label["RESULT.Value"])).abs()
    mae = np.mean(ae)
    # Calculate the Mean Absolute Error (MAE)
    se = (true_label["RESULT.Value"] - predict_label["y_predict"]) ** 2
    mse = np.mean(se)
    # Calculate the Mean Absolute Percentage Error (MAPE)
    ape = abs(true_label["RESULT.Value"] - predict_label["y_predict"]) / true_label["RESULT.Value"] * 100
    mape = np.mean(ape)
    p50_ape = round(ape.quantile(0.5), 4)
    p90_ape = round(ape.quantile(0.9), 4)
    p95_ape = round(ape.quantile(0.95), 4)
    p99_ape = round(ape.quantile(0.99), 4)
    max_ape = round(np.max(ape), 4)
    count_3 = (ape < 3).sum()
    proportion_3 = round(count_3 / len(ape) * 100, 4)

    count_5 = (ape < 5).sum()
    proportion_5 = round(count_5 / len(ape) * 100, 4)
    metric = [predict_label.shape[0], mape, p90_ape, p95_ape, p99_ape, max_ape, proportion_3]
    compare = pd.concat([true_label, predict_label, ape.to_frame(name='APE')], axis = 1)
    return metric, compare

# def merge_K_fold_results(configs):
#     path = configs["K_fold_save_path"]
#     variation_output = 'summary_variation.csv'
#     test_data_output = 'summary_test_data.csv'
#     variation_data = pd.DataFrame()
#     test_results = pd.DataFrame()
#
#     for i in range(configs["n_split"]):
#         folder_path = os.path.join(path, eval(configs["split_folder"])).replace("\\","/")
#         variation_path = os.path.join(folder_path, 'variation.csv')
#         test_data_path = os.path.join(folder_path, 'test_data.csv')
#         variation_split = pd.read_csv(variation_path)
#         test_results_split = pd.read_csv(test_data_path)
#         variation_data = pd.concat([variation_data, variation_split])
#         test_results = pd.concat([test_results, test_results_split])
#     variation_data.to_csv(path + '\\' + variation_output, index=False)
#     test_results.to_csv(path + '\\' + test_data_output, index=False)
#     logger.info(f"merge the {configs['n_split']} Fold results into path {path}")



def save_data_encoder(path, data_encoder):
    save_name = os.path.join(path, "Dada_encoder.joblib").replace("\\", "/")
    dump(data_encoder, save_name)
    logger.info(f"save data_encoder class to {save_name}")





