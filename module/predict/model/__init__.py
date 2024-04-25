#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:03 PM
# @Author  : xiaomanl
# @File    : __init__.py
# @Software: PyCharm

from module.inference.model import lr, fcn, fcn_vec, xgb,base_class
from module.inference.model.lr import LinearModel
from module.inference.model.xgb import  XgboostModel
from module.inference.model.fcn import FCN
from module.inference.model.fcn_vec import FCN_Vec
from module.inference.model.base_class import BaseModel
from module.inference.model.resnet import ResNet
from module.inference.model.atten_resnet import AttenResNet
from module.inference.model.res_trans_net import MultiAttenResNet
from module.inference.model.rf import RandomForest
from module.inference.model.svm import SVMModel
from module.inference.model.lstm import LSTMModel
from module.inference.model.ridge.ridge import Ridge
from module.inference.model.group_multi_atten_resnet import GroupMultiAttenResNet
