#Copyright (C) <2024> Intel Corporation
#SPDX-License-Identifier: Apache-2.0
default_param ={
    "default_numa_per_socket": 1,
    "excel_file": "/home/workspace/benchmark/src/openvino/openvino_emeraldrapids_all_platform_default_param.xlsx",
    "param": {
        "keyword": [
            "batch_size",
            "data_type"
        ],
        "row_bias": 3,
        "column_bias": 3
    },
    "cases_bias": {
        "inference": {
            "throughput": 3,
            "latency": 4,
            "accuracy": 5
        }
    },
    "precision_list": [
        "avx_fp32",
        "amx_int8",
        "amx_bfloat16",
        "amx_fp16",
        "avx_int8"
    ]
}
param_list = {}
for indk, param in enumerate(default_param["param"]["keyword"]):
    param_list[param] = {}
    for function in default_param["cases_bias"]:
        param_list[param][function] = {}
        for mode in default_param["cases_bias"][function]:
            param_list[param][function][mode] = {}
            row = default_param["cases_bias"][function][mode]
            for indp, precision in enumerate(default_param["precision_list"]):
                precision_len = len(default_param["precision_list"])
                column = default_param["param"]["column_bias"] + indk * precision_len + indp
                param_list[param][function][mode][precision] = [row, column]
