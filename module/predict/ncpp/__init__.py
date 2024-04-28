#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:03 PM
# @Author  : xiaomanl
# @File    : __init__.py
# @Software: PyCharm

from module.predict.model import lr, fcn, fcn_vec, xgb,base_class
from module.predict.model.lr import LinearModel
from module.predict.model.xgb import  XgboostModel
from module.predict.model.fcn import FCN
from module.predict.model.fcn_vec import FCN_Vec
from module.predict.model.base_class import BaseModel
from module.predict.model.resnet import ResNet
from module.predict.model.atten_resnet import AttenResNet
from module.predict.model.res_trans_net import MultiAttenResNet
from module.predict.model.rf import RandomForest
from module.predict.model.svm import SVMModel
from module.predict.model.lstm import LSTMModel
from module.predict.model.ridge.ridge import Ridge
from module.predict.model.group_multi_atten_resnet import GroupMultiAttenResNet
