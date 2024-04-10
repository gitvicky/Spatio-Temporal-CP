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

def get_prediction_sets(cal_data, pred_data, alpha, nonconf_score='abs'):
    if nonconf_score == 'abs':
        cal_pred = np.asarray(cal_data[0])
        cal_target = np.asarray(cal_data[1])
        pred_data = np.asarray(pred_data)
        scores = nonconf_score_abs(cal_pred, cal_target)
        n = len(cal_pred)
        qhat = calibrate(scores, n , alpha)
        pred_sets = [pred_data - qhat, pred_data + qhat]
    elif nonconf_score == 'std':
        cal_mean = np.asarray(cal_data[0])
        cal_std = np.asarray(cal_data[1])
        cal_target = np.asarray(cal_data[2])
        pred_mean = np.asarray(pred_data[0])
        pred_std = np.asarray(pred_data[1])
        scores = nonconf_score_lu(cal_target, cal_mean - cal_std, cal_mean + cal_std)
        n = len(cal_mean)
        qhat = calibrate(scores, n , alpha)
        pred_sets = [pred_mean - pred_std - qhat, pred_mean + pred_std + qhat]       
    return pred_sets
    
#Ander's version
def weighted_quantile(scores, alpha, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.quantile(np.sort(scores), alpha, axis = 0, interpolation='higher')
    
    ind=np.argsort(scores, axis=0)
    s=scores[ind]
    w=weights[ind]

    p=1.*w.cumsum()/w.sum()
    y=np.interp(alpha, p, s)

    return y

#Inspired from https://www.pnas.org/doi/abs/10.1073/pnas.2204569119
#Can Handle multi-dimensional outputs
def get_weighted_quantile(scores, quantile, weights):
    
    if weights.ndim == 1:
        weights = weights[:, None]
        scores = scores[:, None]

    #Normalise weights
    p_weights = weights / np.sum(weights, axis=0)

    #Sort the scores and the weights 
    args_sorted_scores = np.argsort(scores, axis=0)
    sortedscores= np.take_along_axis(scores, args_sorted_scores, axis=0)
    sortedp_weights = np.take_along_axis(p_weights, args_sorted_scores, axis=0)

    # locate quantiles of weighted scores per y
    cdf_pweights = np.cumsum(sortedp_weights, axis=0)
    qidx_y = np.sum(cdf_pweights < quantile, axis=0)  # equivalent to [np.searchsorted(cdf_n1, q) for cdf_n1 in cdf_n1xy]
    q_y = sortedscores[(qidx_y, range(qidx_y.size))]
    return q_y