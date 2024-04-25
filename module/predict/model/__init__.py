#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:03 PM
# @Author  : xiaomanl
# @File    : __init__.py
# @Software: PyCharm

from src.inference.model import lr, fcn, fcn_vec, xgb,base_class
from src.inference.model.lr import LinearModel
from src.inference.model.xgb import  XgboostModel
from src.inference.model.fcn import FCN
from src.inference.model.fcn_vec import FCN_Vec
from src.inference.model.base_class import BaseModel
from src.inference.model.resnet import ResNet
from src.inference.model.atten_resnet import AttenResNet
from src.inference.model.res_trans_net import MultiAttenResNet
from src.inference.model.rf import RandomForest
from src.inference.model.svm import SVMModel
from src.inference.model.lstm import LSTMModel
from src.inference.model.ridge.ridge import Ridge
from src.inference.model.group_multi_atten_resnet import GroupMultiAttenResNet
