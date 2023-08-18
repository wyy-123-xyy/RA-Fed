#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w,type_array,local_w_masks,local_b_masks):
    w_avg = copy.deepcopy(w[0])
    all_w_mask, all_b_masks = get_all_masks(type_array,local_w_masks,local_b_masks)

    keys = list(w_avg.keys())
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = w_avg[k] / all_w_mask
    w_avg[k] = torch.div(w_avg[k], all_w_mask)

    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = torch.div(w_avg[k], len(w)) #when pruning weights only
    w_avg[k] = torch.div(w_avg[k] , len(w)) # when pruning weights and bias

    for k in keys[2:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    for e in w_avg:

        w_avg[e] = torch.nan_to_num(w_avg[e], nan = 0)
    return w_avg


def get_all_masks(type_array,local_w_masks,local_b_masks):
    all_w_mask = local_w_masks[0] *0
    all_b_masks = local_b_masks[0] * 0
    for e in type_array:
        all_w_mask += local_w_masks[e - 1]
        all_b_masks += local_b_masks[e - 1]
    return all_w_mask, all_b_masks

#所有参数上传后，相同分区的得进行平均
def FedAvg2(w_glob,w,type_array,local_w_masks,local_b_masks):
    w_avg = copy.deepcopy(w[0])
    all_w_mask, all_b_masks = get_all_masks(type_array,local_w_masks,local_b_masks)
    mask1 = copy.deepcopy(all_w_mask) * 0 + 1
    keys = list(w_avg.keys())
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = w_avg[k] / all_w_mask
    w_avg[k] = torch.where(w_avg[k]==0, w_glob[k], w_avg[k])
    all_w_mask= torch.where(all_w_mask==0, mask1, all_w_mask)
    w_avg[k] = torch.div(w_avg[k], all_w_mask)
#####all_w_mask有对应位置可能为0，这样我们的算法对应位置得换成旧值
    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w)) #when pruning weights only
    # w_avg[k] = torch.div(w_avg[k] , all_b_masks) # when pruning weights and bias

    for k in keys[2:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    '''for e in w_avg:

        w_avg[e] = torch.nan_to_num(w_avg[e], nan = 0)'''
    return w_avg
#所有参数上传后，相同分区的得进行平均
def FedAvg3(w_glob,w,type_array,local_w_masks,local_b_masks):
    w_avg = copy.deepcopy(w[0])
    all_w_mask, all_b_masks = get_all_masks(type_array,local_w_masks,local_b_masks)
    mask1 = copy.deepcopy(all_w_mask) * 0 + 1
    #keys = list(w_avg.keys())
    k = 0
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = w_avg[k] / all_w_mask
    '''w_avg[k] = torch.where(w_avg[k]==0, w_glob[k], w_avg[k])
    all_w_mask= torch.where(all_w_mask==0, mask1, all_w_mask)'''
    w_avg[k] = torch.div(w_avg[k], len(w))
#####all_w_mask有对应位置可能为0，这样我们的算法对应位置得换成旧值
    k = 1
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w)) #when pruning weights only
    # w_avg[k] = torch.div(w_avg[k] , all_b_masks) # when pruning weights and bias

    for k in [2,3]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    '''for e in w_avg:

        w_avg[e] = torch.nan_to_num(w_avg[e], nan = 0)'''
    return w_avg