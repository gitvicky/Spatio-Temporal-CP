#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 February, 2023

@author: vgopakum, agray, lzanisi

Utilities for performing marginal infuctive CP over tensor grids

"""
import numpy as np
import torch
import torch.nn as nn

#Performing the calibration 
def calibrate(scores, n, alpha): 
    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    
#Determining the empirical coverage 
def emp_cov(pred_sets, y_response): 
    return ((y_response >= pred_sets[0]) & (y_response <= pred_sets[1])).mean()

#Estimating the tightness of fit
def est_tight(pred_sets, y_response): #Estimating the tightness of fit
    cov = ((y_response >= pred_sets[0]) & (y_response <= pred_sets[1]))
    cov_idx = cov.nonzero()
    tightness_metric = ((pred_sets[1][cov_idx]  - y_response[cov_idx]) +  (y_response[cov_idx] - pred_sets[0][cov_idx])).mean()
    return tightness_metric

#non-conformity score with lower and upper bars #for both cqr and dropout
def nonconf_score_lu(pred, lower, upper):
    return np.maximum(pred-upper, lower-pred)

#non-conformity score using the absolute error
def nonconf_score_abs(pred, target):
    return np.abs(pred-target)