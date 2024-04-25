#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:03 PM
# @Author  : xiaomanl
# @File    : __init__.py
# @Software: PyCharm

from src.train.model import lr, fcn, fcn_vec, xgb,base_class
from src.train.model.lr import LinearModel
from src.train.model.xgb import  XgboostModel
from src.train.model.fcn import FCN
from src.train.model.fcn_vec import FCN_Vec
from src.train.model.base_class import BaseModel
from src.train.model.resnet import ResNet
from src.train.model.atten_resnet import AttenResNet
from src.train.model.res_trans_net import MultiAttenResNet
from src.train.model.random_forest import RandomForest
from src.train.model.svm import SVMModel
from src.train.model.group_multi_atten_resnet import GroupMultiAttenResNet
from src.train.model.lstm import LSTMModel
from src.train.model.lasso import Lasso
from src.train.model.ridge import Ridge
from src.train.model.elastic_net import ElasticNet

