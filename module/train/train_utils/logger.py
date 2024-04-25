#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:01 PM
# @Author  : xiaomanl
# @File    : logger.py
# @Software: PyCharm
import logging
import yaml
import logging.config
import os
from pathlib import Path

def set_logger(configs,default_path="./logging.yaml", default_level=logging.INFO):
    path = default_path
    root_dir = Path(__file__).resolve().parent
    config_path = os.path.join(root_dir,path).replace("\\", "/")


    try:
        with open(config_path, 'r', encoding="UTF-8") as f:
            logging_config = yaml.load(f, Loader=yaml.FullLoader)
            # change the filename in the file handler
            log_path = configs["output_path"] + "/" + "logs"
            os.makedirs(log_path)
            logging_config["handlers"]["file"]["filename"] = log_path + "/debug.log"
            # change the filename in the error handler
            logging_config["handlers"]["error"]["filename"] = log_path + "/error.log"
            logging.config.dictConfig(logging_config)
    except FileNotFoundError:
        logging.basicConfig(level=default_level)
    logger = logging.getLogger(__name__)


    return logger

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        colors = {
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[31m',
        }
        color = colors.get(record.levelname, '')
        record.levelname = color + record.levelname + '\033[0m'
        record.message = color + record.message + '\033[0m'
        return super().format(record)