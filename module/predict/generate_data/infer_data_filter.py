#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 5:19 PM
# @Author  : xiaomanl
# @File    : data_filter.py
# @Software: PyCharm
import os.path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from sklearn.model_selection import GroupKFold

from src.inference.utils import calculate_running_time, read_file, generate_abs_path, read_yaml_file, read_class_config
from src.inference.utils import multi_label_transfer, linpack_transfer, MLC_multi_label_transfer
import re
import logging
from pathlib import Path
import ast
import numpy as np
import sys



class DataLoader():
    def __init__(self, configs, K_Fold=False, config_file="data_filter.yaml"):
        """
        Initialize the parameters

        Parameters:
        args (dict): Dictionary of arguments
        configs (dict): Dictionary of configuration
        """
        self.configs = configs
        self.root_dir = configs['root_dir']
        self.configs.update(read_class_config(Path(__file__).resolve().parent, config_file))
        self.logger = logging.getLogger("DataLoader")

        self.get_platform_path()
        self.output_path = generate_abs_path(self.root_dir, self.configs["output_path"])
        self.select_model = self.configs["select_model"]
        self.label_name_list = self.configs["label_name_list"]
        self.use_multiple_label = self.configs["use_multiple_label"]
        self.feature_config_name = f"{self.select_model}_feature_config"
        self.stream_name_mapping_dict = self.configs["replace_dict"]
        self.workload_name = self.configs["workload_names"][0]
        self.feature_config = self.configs[self.feature_config_name]
        self.new_feature_rule = self.configs["new_feature_rule"]


        self.prefix = 'QDF.'
        self.qdf_features = self.configs["qdf_features"]
        # self.generate_test_data = self.configs["generate_test_data"]
        self.col_filter_nan_row = self.configs["col_filter_nan_row"]
        self.norm_feature_col_mapping = self.configs["norm_feature_col_mapping"]
        self.add_mem_features = self.configs["add_mem_features"]
        self.test = {}

    def get_platform_path(self):
        if "platform" not in self.configs.keys():
            self.configs["platform"] = "SPR"

        if self.configs["platform"] == "SPR":
            qdf_data_path = self.configs["infer_spr_qdf_data_path"]
            dataset_path = self.configs["infer_spr_dataset_path"]
            self.qdf_path = generate_abs_path(self.root_dir, qdf_data_path)
            self.data_path = generate_abs_path(self.root_dir, dataset_path)
        elif self.configs["platform"] == "EMR":
            qdf_data_path = self.configs["infer_emr_qdf_data_path"]
            dataset_path = self.configs["infer_emr_dataset_path"]
            self.qdf_path = generate_abs_path(self.root_dir, qdf_data_path)
            self.data_path = generate_abs_path(self.root_dir, dataset_path)

    def split_feature(self, feature_name, data):
        data[feature_name] = data[feature_name].str.split()

        # Create new columns based on the split values
        for value in data[feature_name]:
            for item in value:
                column_name, column_value = item.split(':')
                new_column_name = f'{feature_name}.{column_name}'
                data[new_column_name] = column_value
                self.configs[self.feature_config_name][new_column_name] = {
                    'used_in_training': 1,
                    'processing_method': {'name': 'Onehot_encoding'},
                    'data_type': 'float64'
                }
        return data

    def linpack_add_feature(self, feature_name, data):
        if not feature_name:
            error_message = f"Error: {feature_name} is None or empty in linpack_add_feature."
            self.logger.error(error_message)
            return data
        data[feature_name] = np.where(data['N_SIZE'].astype(float) == 120000, 0, 1)
        data[feature_name] = data[feature_name].astype(int)
        self.configs[self.feature_config_name][feature_name] = {
            'used_in_training': 1,
            'processing_method': {'name': 'Onehot_encoding'},
            'data_type': 'float64'
        }
        return data

    def add_rank_feature(self, feature_name, data):
        if not feature_name:
            error_message = f"Error: {feature_name} is None or empty in add_rank_feature."
            self.logger.error(error_message)
            return data
        data[feature_name] = data["Measure.DIMM.PartNo"].str.lstrip('|').str.split('|').str[0].map(self.configs["Dimm_population_to_rank_mapping_dict"])

        not_found_values = data[data[feature_name].isnull()]["Measure.DIMM.PartNo"].unique()
        self.logger.warning(f"memory partno not found in the dictionary:{not_found_values}")
        data[feature_name] = (data[feature_name].map(self.configs["rank_to_number_dict"])).fillna(0)
        data[feature_name] = data[feature_name].astype(int)
        self.configs[self.feature_config_name][feature_name] = {
            'used_in_training': 1,
            'processing_method': {'name': 'Normalization_scaler'},
            'data_type': 'float64'
        }
        return data

    def multiply_feature(self, feature_name, data):
        if not feature_name:
            error_message = f"Error: {feature_name} is None or empty in multiply_feature."
            self.logger.error(error_message)
            return data
        # Remove the parentheses from the string
        string = feature_name.strip("()")

        # Split the string by comma and strip any extra whitespace
        feature_name_list = [value.strip() for value in string.split(",")]

        feature_new_name = "*".join(feature_name_list)
        if feature_new_name in self.norm_feature_col_mapping:
            feature_new_name = self.norm_feature_col_mapping[feature_new_name]
        data[feature_new_name] = data[feature_name_list[0]]
        try:
            for col in feature_name_list[1:]:
                data[feature_new_name] = data[feature_new_name].astype(float) * data[col].astype(float)
        except ValueError:
            error_message = f"Error: One or more values in {feature_name_list}  cannot be converted to float."
            self.logger.error(error_message)
        self.configs[self.feature_config_name][feature_new_name] = {
            'used_in_training': 1,
            'processing_method': {'name': 'Normalization_scaler'},
            'data_type': 'float64'
        }
        return data

    def divide_feature(self, feature_name, data):

        # Remove the parentheses from the string
        string = feature_name.strip("()")

        # Split the string by comma and strip any extra whitespace
        feature_name_list = [value.strip() for value in string.split(",")]

        feature_new_name = "--".join(feature_name_list)
        data[feature_new_name] = data[feature_name_list[0]]
        try:
            for col in feature_name_list[1:]:
                data[feature_new_name] = data[feature_new_name].astype(float) / data[col].astype(float)
        except ValueError:
            error_message = f"Error: One or more values in {feature_name_list}  cannot be converted to float."
            self.logger.error(error_message)

        feature_config_name = f"{self.select_model}_feature_config"
        self.configs[feature_config_name][feature_new_name] = {
            'used_in_training': 1,
            'processing_method': {'name': 'Normalization_scaler'},
            'data_type': 'float64'
        }
        return data


    def data_filter(self, merge_data):
        """
        Filter the data by removing the rows with NaN values, equal values and unneeded workload names.

        Parameters:
        merge_data (pd.DataFrame): Merged data from the raw data and QDF data

        Returns:
        pd.DataFrame: Filtered data
        """
        self.logger.info("Begin filtering the data")

        # for key, value in self.configs["data_value_filter"].items():
        #     merge_data = merge_data[~merge_data[key].isin(value)]


        merge_data.loc[(merge_data['RESULT.TestName'].str.contains('Bandwidth')) & (merge_data[
                                                                                      'RESULT.WorkloadName'] == 'Memory Latency Checker'), 'RESULT.WorkloadName'] += ' Bandwidth'
        merge_data.loc[(merge_data['RESULT.TestName'].str.contains('latency')) & (merge_data[
                                                                                    'RESULT.WorkloadName'] == 'Memory Latency Checker'), 'RESULT.WorkloadName'] += ' latency'
        for key, value in self.configs["data_value_filter"].items():
            merge_data = merge_data[~merge_data[key].isin(value)]
        if "All Workloads" not in self.configs["workload_names"]:
            # assert
            merge_data = merge_data[merge_data["RESULT.WorkloadName"].isin(self.configs["workload_names"])]
        # merge_data = merge_data[merge_data["RESULT.Value"] > 200]
        if "All Testcases" not in self.configs["test_names"]:
            merge_data = merge_data[merge_data["RESULT.TestName"].isin(self.configs["test_names"])]
        # merge_data = merge_data[~merge_data["RESULT.TestName"].str.contains('Latency|latency', case=False, na=False)]

        self.logger.info(
            f"After filter, data has \033[1;34;34m{merge_data.shape[0]}\033[0m rows and \033[1;34;34m{merge_data.shape[1]}\033[0m columns.")
        # merge_data = merge_data.dropna(subset=self.col_filter_nan_row)
        # self.logger.info(
        #     f"After dropna, data has \033[1;34;34m{merge_data.shape[0]}\033[0m rows and \033[1;34;34m{merge_data.shape[1]}\033[0m columns.")

        merge_data = merge_data.reset_index(drop=True)
        self.logger.info(
            f"After delete iteration variation > 10%, data has \033[1;34;34m{merge_data.shape[0]}\033[0m rows and \033[1;34;34m{merge_data.shape[1]}\033[0m columns.")



        return merge_data

    def extract_number(self, string):
        match = re.search(r'\d+(?:\.\d+)?', string)
        return float(match.group(0)) if '.' in match.group(0) else int(match.group(0))

    def detect_duplicate_columns(self, merge_data):
        duplicate_columns = set()
        columns = set(merge_data.columns)
        for column in columns:
            for compare_column in columns:
                if column != compare_column and (merge_data[column] == merge_data[compare_column]).all():
                    duplicate_columns.add(tuple(sorted((column, compare_column))))
        return duplicate_columns

    def train_infer_data_split(self, train_inds, test_inds, merge_data):
        for i in range(self.configs["n_split"]):
            train_validation_dataset = merge_data.iloc[train_inds[i]]
            infer_dataset = merge_data.iloc[test_inds[i]]
            self.logger.info(train_validation_dataset.describe())
            self.logger.info(infer_dataset.describe())
            train_validation_dataset.to_csv(f"./data_split/train_{i}.csv", index=False)
            infer_dataset.to_csv(f"./data_split/infer_{i}.csv", index=False)

    def add_qdf_details(self, filter_data):
        self.logger.info("Begin merge QDF infomation!")
        qdf_information = read_file(self.qdf_path).dropna(axis=0, subset=['QDF/SSPEC'])
        # raw_data_qdf_values = set(raw_data["META.metadata.cscope.qdf0"].values)
        # qdf_information = qdf_information[qdf_information["QDF/SSPEC"].isin(raw_data_qdf_values)]
        qdf_information = qdf_information.add_prefix(self.prefix)
        self.logger.info(
            f"QDF raw data has \033[1;34;34m{qdf_information.shape[0]}\033[0m rows and \033[1;34;34m{qdf_information.shape[1]}\033[0m columns.")

        qdf_information = qdf_information[self.qdf_features]


        self.logger.info(
            f"selected QDF raw data has \033[1;34;34m{qdf_information.shape[0]}\033[0m rows and \033[1;34;34m{qdf_information.shape[1]}\033[0m columns.")

        test_data = pd.merge(filter_data, qdf_information, left_on='META.metadata.cscope.qdf0',
                             right_on=self.prefix + 'QDF/SSPEC', how='left')
        # , $indicator=True
        test_data["_merge"] = test_data[self.prefix + "QDF/SSPEC"].isna().map({True: "left_only", False: "both"})
        no_match_data = test_data[test_data["_merge"] == "left_only"]
        no_match_qdf = set(no_match_data["META.metadata.cscope.qdf0"].values)
        self.logger.warning(f"These QDF cannot be found details in file {self.qdf_path}:\n{list(no_match_qdf)}")

        merge_data_before = pd.merge(filter_data, qdf_information, left_on='META.metadata.cscope.qdf0',
                              right_on=self.prefix + 'QDF/SSPEC')

        merge_data = merge_data_before.dropna(subset=["Measure.DIMM.PartNo"])
        merge_data = merge_data.dropna(axis=1)
        # merge_data = merge_data_before.drop(
        #     merge_data_before[(merge_data_before['QDF.AVX3 Sngl Core Turbo Freq Rate'] == 0)].index)
        drop_columns = [item for item in merge_data_before.columns if item not in merge_data.columns]
        self.logger.error(f"drop columns: {drop_columns}")
        # merge_data = merge_data.drop(columns=['META.metadata.cscope.qdf0'])
        merge_data[self.prefix + 'Thermal Design Power'] = merge_data[self.prefix + 'Thermal Design Power'].apply(
            self.extract_number)
        merge_data[self.prefix + 'Speed'] = merge_data[self.prefix + 'Speed'].apply(self.extract_number)
        merge_data[self.prefix + 'Cache'] = merge_data[self.prefix + 'Cache'].apply(self.extract_number)
        merge_data[self.prefix + 'Max Turbo Frequency Rate'] = merge_data[self.prefix + 'Max Turbo Frequency Rate'].apply(self.extract_number)

        if "Stream" in self.configs["workload_names"]:

            merge_data['RESULT.TestName'] = merge_data['RESULT.TestName'].replace(self.stream_name_mapping_dict)

        # merge_data = merge_data.drop(columns=['QDF/SSPEC'])
        duplicate_columns = self.detect_duplicate_columns(merge_data)
        delete_columns = [item[1] for item in duplicate_columns]
        self.logger.warning(f"Duplicated columns: {duplicate_columns}")
        # merge_data = merge_data.drop(columns=delete_columns)
        self.logger.info(
            f"After merge QDF, it has \033[1;34;34m{merge_data.shape[0]}\033[0m rows and \033[1;34;34m{merge_data.shape[1]}\033[0m columns.")


        ## drop these feature
        origin_qdf = ["Cache", "Speed", "Functional Core", "DDR5 FREQ"]
        # origin_qdf = ["Cache", "Speed", "Functional Core", "DDR5 FREQ", "MAXIMUM PEG LANE COUNT",
        #               "TMUL Sngl Core Turbo Freq Rate", "AVX3 Sngl Core Turbo Freq Rate",
        #               "AVX2 Sngl Core Turbo Freq Rate",
        #               "AVX Deterministic P1 Freq Rte"]
        drop_qdf = [self.prefix + item for item in origin_qdf]

        # Delete columns that have 'ISS' in their name
        iss_cols = [col for col in merge_data.columns if 'ISS' in col]
        merge_data = merge_data.drop(columns=iss_cols)
        column_list = merge_data.columns.tolist()
        intersection = list(set(drop_qdf) & set(column_list))
        merge_data = merge_data.drop(columns=intersection)
        merge_data.to_csv(self.output_path + "/origin_data_add_qdf.csv", index=False)
        self.logger.info(
            f"merge_data has \033[1;34;34m{merge_data.shape[0]}\033[0m rows and \033[1;34;34m{merge_data.shape[1]}\033[0m columns.")

        return merge_data

    def samsang_part_num_decode(self, part_num):
        part_num = part_num.split('|')[0]
        csv_list = pd.read_csv(self.root_dir + "/../../data/processed/samsang_dimm.csv")
        rank = self.configs["Dimm_population_to_rank_mapping_dict"].get(part_num, "Dual Rank")
        decode_info = {
            "DIMM Type": None,
            "DDR": None,
            "Density": None,
            "Organization": None,
            "Rank": {"Dual Rank": 2, "Quad Rank": 4, "Single Rank": 1}.get(rank, 2),
            "Speed": None,
            "CL": 40

        }
        part_num = part_num[:-2]
        # print(part_num)
        row = csv_list[csv_list["Part Number"].str.strip() == part_num]
        # print(row.shape)
        # M329R4GA0BB0-CQK
        if row.empty:
            self.logger.warning(f"Part number {part_num} not found in the samsang_dimm csv file.")
        else:
            # self.logger.debug(f"Part number {part_num} found in the samsang_dimm csv file.")
            strs = row["Rank x Organization"].values[0]
            Rank, Organization  = re.findall(r'(\d)R x\s?(\d+)', strs)[0]
            decode_info = {
                "DIMM Type": row["Dimm Type"].values[0],
                "DDR": row["DDR"].values[0],
                "Density": self.extract_number(row["Density"].values[0]),
                "Organization": int(Organization),
                "Rank": int(Rank),
                "Speed": self.extract_number(row["Speed"].values[0]),
                "CL": 40

            }

        return decode_info

    def hynix_part_num_decode(self, part_number):
        rank = self.configs["Dimm_population_to_rank_mapping_dict"].get(part_number, "Dual Rank")
        decode_info = {
            "DIMM Type": None,
            "DDR": None,
            "Density": None,
            "Organization": None,
            "Rank": {"Dual Rank": 2, "Quad Rank": 4, "Single Rank": 1}.get(rank, 2),
            "Speed": None,
            "CL": None

        }
        part_number = part_number.split('|')[0]
        product_density = {
            'G6': '8GB', 'G7': '16GB', 'G8': '32GB', 'G9': '64GB',
            'T0': '128GB', 'T1': '256GB', 'T2': '512GB'
        }
        organization = {'4': 'X4', '8': 'X8', '6': 'X16'}
        # generation = {'M': '1st', 'A': '2nd', 'B': '3rd', 'C': '4th'}
        module_type = {
            'U': 'UDIMM', 'S': 'SODIMM', 'E': 'ECC UDIMM',
            'A': 'ECC SODIMM', 'R': 'RDIMM', 'Q': 'RDIMM (x72)', 'L': 'LRDIMM'
        }
        speed_dict = {'EB': 4800, 'GB': 5600, 'EE': 4800, 'GE': 5600, 'HB': 6400, 'HE': 6400}
        CL_dict = {'EB': 40, 'GB': 46, 'EE': 46, 'GE': 52, 'HB': 52, 'HE': 60}

        # Parse the part number
        try:
            rank = self.configs["Dimm_population_to_rank_mapping_dict"].get(part_number, "Dual Rank")
            decode_info["Rank"] = {"Dual Rank": 2, "Quad Rank": 4, "Single Rank": 1}.get(rank, 2)
            decode_info['DDR'] = "DDR5" if part_number[0:3] else "Unknown"
            density = product_density.get(part_number[3:5], "0 Density")
            decode_info['Density'] = self.extract_number(density)
            decode_info['Organization'] = int(part_number[5])

            speed = part_number[7:9]  # Speed code, more complex to decode
            decode_info["Speed"] = speed_dict.get(speed, 0)
            mod_type = module_type.get(part_number[9], "Unknown")
            decode_info['DIMM Type'] = mod_type
            CL = CL_dict.get(speed, 40)
            decode_info['CL'] = CL
        except:
            self.logger.warning(f"{part_number} Invalid part number format")

        return decode_info

    def micro_part_num_decode(self, part_number):
        rank = self.configs["Dimm_population_to_rank_mapping_dict"].get(part_number, "Dual Rank")
        decode_info = {
            "DIMM Type": None,
            "DDR": None,
            "Density": None,
            "Organization": None,
            "Rank": {"Dual Rank": 2, "Quad Rank": 4, "Single Rank": 1}.get(rank, 2),
            "Speed": None,
            "CL": None

        }
        part_number = part_number.split('|')[0]
        # Define the regular expression pattern to match the part number
        pattern = r"MT(\w)(\d){2}(\w)(\d)(\d)(\d)(\w)(\w)(\d)(\w)(\w)(\d{2}\w)(\w)(\w)"
        match = re.match(pattern, part_number)

        if match is None:
            self.logger.warning(f"{part_number} Invalid part number format")
            return decode_info

        # Extract the components of the part number
        product_family, die_number, Voltage, package_ranks, logical_ranks, component_config, module_density, \
            module_height, die_in_package, module_type, temperature_range,speed_bin, revision_code, designator = match.groups()

        # Convert the matched groups to their corresponding values using the provided tables
        module_type_dict = {
                            "S": "SODIMM X64",
                            "V": "CSODIMM X64",
                            "T": "SOEDIMM X72",
                            "W": "CSODIMM X72",
                            "U": "UDIMM X64",
                            "A": "CUDIMM X64",
                            "E": "EUDIMM X72",
                            "B": "CUDIMM X72",
                            "R": "RDIMM X80",
                            "P": "RDIMM X72"
                        }
        module_density_dict = {'3': '8GB','Z': '12GB','4': '16GB','Y': '24GB','5': '32GB','X': '48GB','6': '64GB','W': '96GB',
                               '7': '128GB','V': '192GB','8': '256GB','9': '512GB','A': '1024GB','B': '2048GB','C': '4096GB'}

        speed_bin_dict  =  {
                '32B': (3200, '26'),
                '36B': (3600, '30'),
                '40B': (4000, '32'),
                '44B': (4400, '36'),
                '48B': (4800, '40'),
                '52B': (5200, '42'),
                '56B': (5600, '46'),
                '60B': (6000, '48'),
                '64B': (6400, '52'),
                '68B': (6800, '56'),
                '72B': (7200, '58'),
                '76B': (7600, '62'),
                '80B': (8000, '64'),
                '88B': (8800, '72')
            } # Continue for all speed bin codes
        rank = self.configs["Dimm_population_to_rank_mapping_dict"].get(part_number, "Dual Rank")
        speed, cl = speed_bin_dict.get(speed_bin)

        decode_info = {
            "DIMM Type": module_type_dict.get(module_type, "Unknown"),
            "DDR": 'DDR5' if part_number[2] == "C" else 'Unknown',
            "Density": self.extract_number(module_density_dict.get(module_density, "Unknown")),
            "Organization": int(component_config),
            "Rank": int(package_ranks),
            "Speed": int(speed),
            "CL": int(cl)

        }
        return decode_info

    def add_feature_process_config(self, feature_new_name, processing_method, data_type):
        self.configs[self.feature_config_name][feature_new_name] = {
            'used_in_training': 1,
            'processing_method': {'name': processing_method},
            'data_type': data_type
        }

    def decode_dimm_part_number(self, part_number):
        decode_info = {
            "DIMM Type": None,
            "DDR": None,
            "Density": None,
            "Organization": None,
            "Rank": None,
            "Speed": None,
            "CL": None

        }
        if not part_number:
            error_message = f"Error: part_number is None or empty."
            self.logger.warning(error_message)
        else:
            if part_number.startswith("HMC"):
                decode_info = self.hynix_part_num_decode(part_number)

            elif part_number.startswith("MTC"):
                decode_info = self.micro_part_num_decode(part_number)

            elif part_number.startswith("M3"):
                decode_info = self.samsang_part_num_decode(part_number)

        return decode_info

    def preset_processing(self, value):
        if "default" or "SSE" in value:
            result =  "SSE"
        else:
            tranfer_value = value.split("|")
            if len(tranfer_value) == 1:
                result = tranfer_value[0]
            else:
                result =  tranfer_value[1]
        self.test[value] = result
        return result

    def processing_cache(self, coll1d, coll1i, coll2, coll3):
        r_coll1d, r_coll1i, r_coll2, r_coll3 = coll1d, coll1i, coll2, coll3
        if coll1d < 10:
            r_coll1d =  coll1d * 1024
        if coll1i < 10:
            r_coll1i =  coll1i * 1024
        if coll2 < 1000:
            r_coll2 =  coll2 * 1024
        if coll3 < 1000:
            r_coll3 =  coll3 * 1024
        return r_coll1d, r_coll1i, r_coll2, r_coll3


    @calculate_running_time
    def run(self, K_Fold = False):

        self.logger.info(f"Begin reading raw data from {self.data_path}")
        raw_data = read_file(self.data_path)

        self.logger.info(f"Raw data has \033[1;34;34m{raw_data.shape[0]}\033[0m rows and \033[1;34;34m{raw_data.shape[1]}\033[0m columns.")

        filter_data = self.data_filter(raw_data)
        assert filter_data is not None

        filter_data["RESULT.Value"] = filter_data[["RESULT.Value"]]
        self.logger.info(f"filtered data has \033[1;34;34m{filter_data.shape[0]}\033[0m rows and \033[1;34;34m{filter_data.shape[1]}\033[0m columns.")

        test_names = raw_data["RESULT.TestName"].unique()
        test_names_str = "\n".join(test_names).center(80)
        self.logger.debug(f'List of testnames:{test_names_str}')
        # get the set of column names in raw_data

        #
        if self.configs["add_qdf_details"]:
            raw_columns = set(raw_data.columns)
            # get the set of column names in merge_data
            merge_data = self.add_qdf_details(filter_data)
            merge_columns = set(merge_data.columns)
            # get the set of column names that are in merge_data but not in raw_data
            missing_columns = merge_columns.difference(raw_columns)
            merge_data.to_csv(self.output_path + "/merge_data_raw.csv", index = False)
            self.logger.debug(f"QDF colums list: {missing_columns}")
        else:
            merge_data = filter_data

        # merge_data = filter_data


        merge_data = merge_data.drop(merge_data.filter(like='@timestamp').columns, axis=1)
        if "RESULT.WorkloadPreset" in merge_data.columns:
            merge_data["RESULT.WorkloadPreset"] = merge_data["RESULT.WorkloadPreset"].apply(self.preset_processing)

        if self.use_multiple_label:
            if "Linpack" in self.configs["workload_names"]:
                transfer_merge_data, unfold_col_name_list = linpack_transfer(merge_data, "RESULT.TestName", "RESULT.Value")
            else:
                transfer_merge_data, unfold_col_name_list = multi_label_transfer(merge_data, "RESULT.TestName", "RESULT.Value")
            self.configs["label_name_list"] = unfold_col_name_list
        else:
            unfold_col_name_list = self.label_name_list
            transfer_merge_data = merge_data
        merge_data = transfer_merge_data


        for operator in self.new_feature_rule:
            operate_features = self.new_feature_rule[operator]
            if operate_features is None:
                continue
            if operator == "ADD":
                for operate_feature in operate_features:
                    if operate_feature == "DIMM.RANK":
                        merge_data = self.add_rank_feature(operate_feature, merge_data)
                    elif operate_feature == "N_SIZE_RANGE" and "Linpack" in self.configs["workload_names"]:
                        merge_data = self.linpack_add_feature(operate_feature, merge_data)
                    else:
                        self.logger.error(f"Not support {operate_feature}")
            elif operator == "MULTIPLY":
                for operate_feature in operate_features:
                    merge_data = self.multiply_feature(operate_feature, merge_data)
            elif operator == "DIVIDE":
                for operate_feature in operate_features:
                    merge_data = self.divide_feature(operate_feature, merge_data)
            elif operator == "SPLIT":
                for operate_feature in operate_features:
                    merge_data = self.split_feature(operate_feature, merge_data)
        if self.add_mem_features:
            new_features = pd.DataFrame(merge_data["Measure.DIMM.PartNo"].apply(self.decode_dimm_part_number).tolist())
            merge_data = pd.concat([merge_data, new_features], axis=1)

        if "RESULT.WorkloadPreset" in merge_data.columns:
            merge_data["RESULT.WorkloadPreset"] = merge_data["RESULT.WorkloadPreset"].apply(self.preset_processing)
        self.logger.info(self.test)

        header = pd.DataFrame(merge_data.columns)
        header = header.T
        header.to_csv(self.output_path + "/feature_header.csv", index=False, header=False)
        assert merge_data is not None
        # merge_data.loc[merge_data["RESULT.TestName"].str.contains('bandwidth'), 'a'] += 'bandwidth'

        # merge_data[
        #     ["SVR.CPU.L1d Cache", "SVR.CPU.L1i Cache", "SVR.CPU.L2 Cache", "SVR.CPU.L3 Cache"]] = merge_data.apply(
        #     lambda x: self.processing_cache(x["SVR.CPU.L1d Cache"], x["SVR.CPU.L1i Cache"], x["SVR.CPU.L2 Cache"],
        #                                     x["SVR.CPU.L3 Cache"]), axis=1, result_type="expand")

        all_label = merge_data[unfold_col_name_list]
        all_features = merge_data.drop(columns=unfold_col_name_list)
        all_features = all_features.rename(columns=self.norm_feature_col_mapping)





        self.logger.info(f"The shape of features data is: \033[1;34;34m{all_features.shape[0]}\033[0m rows and \033[1;34;34m{all_features.shape[1]}\033[0m columns.")
        self.logger.info(f"The shape of label data is: \033[1;34;34m{all_label.shape[0]}\033[0m rows and \033[1;34;34m{all_label.shape[1]}\033[0m columns.")

        # self.logger.debug(merge_data.dtypes)
        self.logger.info("Workload list: {}".format(", ".join(filter_data["RESULT.WorkloadName"].unique())))
        # string_data.to_csv(f'{self.configs["save_path"]}/string.csv', index=False)
        return all_features, all_label, merge_data


