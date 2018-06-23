#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:11:28 2018

@author: hutongxin
"""


import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.autograd import Variable

import faster_rcnn.additional_model.gmm
from faster_rcnn.additional_model.gmm import apply_gmm

import faster_rcnn.network
from faster_rcnn.network import FC, Conv2d


class GaussianMixtureModel(Module):
    def __init__(self, input, nhidden):
        super(GaussianMixtureModel, self).__init__()

        self.fc_gmm = FC(input, 2 * nhidden, relu=True)
        faster_rcnn.network.weights_normal_init(self.fc_gmm, 0.01)

    def forward(self, rois_1, rois_2, r_feature, dropout, training):
        """
        :param rois_1: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param rois_2: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param r_feature Variable size: Batch * 25088
        :return: cat_feature Variable size: Batch * 1024
        """
        pairs = torch.cat((rois_1.data[:, 1:], rois_2.data[:, 1:]), 1)  # [batch,8](xs1,ys1,xs2,ys2,xo1,yo1,xo2,yo2)
        pairs = pairs.cpu().numpy()
        spatial_vector = apply_gmm(pairs, r"/home/hutong/tmp/xin/MSDN/tools/", r"gmmmodel_10000randnum_3000pic.m")
        spatial_vector = faster_rcnn.network.np_to_variable(spatial_vector, is_cuda=True)  # [batch,400]

        union_feature = torch.cat((r_feature, spatial_vector), 1)  # [batch,25488]
        cat_feature = self.fc_gmm(union_feature)  # [batch, 1024]
        if dropout:
            cat_feature = F.dropout(cat_feature, training=training)

        return cat_feature


class DualMask(Module):
    def __init__(self, nhidden):
        super(DualMask, self).__init__()
        self.conv1 = Conv2d(2, 96, kernel_size=5)
        self.conv2 = Conv2d(96, 128, kernel_size=5)
        self.conv3 = Conv2d(128, 64, kernel_size=8)
        self.fc1_dm = FC(18496, nhidden, relu=True)
        self.fc2_dm = FC(25600, 2 * nhidden, relu=True)
        faster_rcnn.network.weights_normal_init(self.fc1_dm, 0.01)
        faster_rcnn.network.weights_normal_init(self.fc2_dm, 0.01)

    def forward(self, im_info, rois_1, rois_2, r_feature, dropout, training):
        """
        :param im_info: Variable, size: 1 * 3
        :param rois_1: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param rois_2: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param r_feature Variable size: Batch * 25088
        :return: cat_feature Variable size: Batch * 1024
        """
        mask1 = get_dual_mask(im_info, rois_1)  # [batch,32,32]
        mask2 = get_dual_mask(im_info, rois_2)
        mask_pairs = np.zeros((mask1.shape[0], 2, mask1.shape[1], mask1.shape[2]))
        mask_pairs[:, 0, :, :] = mask1
        mask_pairs[:, 1, :, :] = mask2
        mask_pairs = faster_rcnn.network.np_to_variable(mask_pairs, is_cuda=True)  # [batch,2,32,32]

        mask_pairs = self.conv1(mask_pairs)  # [batch,96,28,28]
        mask_pairs = self.conv2(mask_pairs)  # [batch,128,24,24]
        mask_pairs = self.conv3(mask_pairs)  # [batch,64,17,17]

        spatial_feature = mask_pairs.contiguous().view(mask_pairs.size()[0], -1)  # [batch,18496]
        spatial_feature = self.fc1_dm(spatial_feature)  # [batch, 512]
        if dropout:
            spatial_feature = F.dropout(spatial_feature, training=training)

        cat_feature = torch.cat((r_feature, spatial_feature), 1)  # [batch,25600]
        cat_feature = self.fc2_dm(cat_feature)  # [batch, 1024]
        if dropout:
            cat_feature = F.dropout(cat_feature, training=training)

        return cat_feature


class GeometricSpatialFeature(Module):
    def __init__(self, nhidden, dropout):
        super(GeometricSpatialFeature, self).__init__()
        self.fc = FC(640, nhidden, relu=True)
        self.fc_r = FC(640, nhidden, relu=True)
        self.dropout = dropout

    def forward(self, rois_s, rois_o, feature_s, feature_o, feature_r):
        """
        :param rois_s: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param rois_o: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param feature_s Variable size: Batch * 512
        :param feature_o Variable size: Batch * 512
        :param feature_r Variable size: Batch * 512
        :return: cat_s Variable size: Batch * 512
        :return: cat_o Variable size: Batch * 512
        :return: cat_r Variable size: Batch * 512
        """

        spatial_s = self.convert_object(rois_s)   # [batch, 128]
        spatial_o = self.convert_object(rois_o)
        spatial_r = self.convert_relation(rois_s, rois_o)
        cat1 = torch.cat((feature_s, spatial_s), dim=1)   # [batch, 640]
        cat2 = torch.cat((feature_o, spatial_o), dim=1)
        cat3 = torch.cat((feature_r, spatial_r), dim=1)
        cat_s = self.fc(cat1)
        if self.dropout:
            cat_s = F.dropout(cat_s, training=self.training)
        cat_o = self.fc(cat2)
        if self.dropout:
            cat_o = F.dropout(cat_o, training=self.training)
        cat_r = self.fc_r(cat3)
        if self.dropout:
            cat_r = F.dropout(cat_r, training=self.training)

        return cat_s, cat_o, cat_r

    def convert_object(self, rois, dim=32):
        """
        :param rois: Variable, size: Batch * 5 (0, x1, y1, x2, y2)
        :param dim: int
        :return: Batch_Size x 4dim
        """
        x = rois[:, 1:2]
        y = rois[:, 2:3]
        w = rois[:, 3:4] - rois[:, 1:2]
        h = rois[:, 4:5] - rois[:, 2:3]

        highD = self.convert([x, y, w, h], dim=dim)

        return highD

    def convert_relation(self, rois_1, rois_2, dim=32):
        x_s = rois_1[:, 1:2]
        y_s = rois_1[:, 2:3]
        w_s = rois_1[:, 3:4] - rois_1[:, 1:2]
        h_s = rois_1[:, 4:5] - rois_1[:, 2:3]

        x_o = rois_2[:, 1:2]
        y_o = rois_2[:, 2:3]
        w_o = rois_2[:, 3:4] - rois_2[:, 1:2]
        h_o = rois_2[:, 4:5] - rois_2[:, 2:3]

        a = torch.abs(x_s - x_o)/w_o
        b = torch.abs(y_s - y_o)/h_o
        c = torch.log(w_s/w_o)
        d = torch.log(h_s/h_o)

        highD = self.convert([a, b, c, d], dim=dim)

        return highD

    @staticmethod
    def convert(position_info, dim):
        assert dim % 2 == 0

        denom = position_info[0].data.new(dim/2).fill_(10000)
        denom = Variable(torch.pow(denom, torch.linspace(0, dim, dim/2 + 1)[:-1].type_as(denom) / dim))

        output = []
        for x in position_info:
            output.append(torch.sin(x / denom))
            output.append(torch.cos(x / denom))

        return torch.cat(output, dim=1)


def get_dual_mask(im_info, bb):
    bb = bb[:, 1:].cpu().data.numpy()  # [batch,4]
    ih = im_info[0][0]
    iw = im_info[0][1]
    rh = 32.0 / ih
    rw = 32.0 / iw
    x1 = np.maximum(0, (np.floor(bb[:, 0] * rw)).astype(int))  # [batch,]
    x2 = np.minimum(32, (np.ceil(bb[:, 2] * rw)).astype(int))
    y1 = np.maximum(0, (np.floor(bb[:, 1] * rh)).astype(int))
    y2 = np.minimum(32, (np.ceil(bb[:, 3] * rh)).astype(int))
    mask = np.zeros((x1.shape[0], 32, 32))  # [batch,32,32]
    batch = x1.shape[0]
    for i in range(batch):
        mask[i, y1[i]: y2[i], x1[i]: x2[i]] = 1
        assert (mask[i].sum() == (y2[i] - y1[i]) * (x2[i] - x1[i]))
    return mask


def remove_bg_info(features, bb1, bb2):
    bb1 = bb1[:, 1:].cpu().data.numpy()  # [batch,4] (x1,y1,x2,y2)
    bb2 = bb2[:, 1:].cpu().data.numpy()
    features = features.cpu().data.numpy()  #[1,512,H,W]
    H = features.shape[2]
    W = features.shape[3]
    xs1 = np.maximum(0, (np.floor(bb1[:, 0] / 16.0)).astype(int))  # [batch,]
    xs2 = np.minimum(W, (np.ceil(bb1[:, 2] / 16.0)).astype(int))
    ys1 = np.maximum(0, (np.floor(bb1[:, 1] / 16.0)).astype(int))
    ys2 = np.minimum(H, (np.ceil(bb1[:, 3] / 16.0)).astype(int))
    xo1 = np.maximum(0, (np.floor(bb2[:, 0] / 16.0)).astype(int))
    xo2 = np.minimum(W, (np.ceil(bb2[:, 2] / 16.0)).astype(int))
    yo1 = np.maximum(0, (np.floor(bb2[:, 1] / 16.0)).astype(int))
    yo2 = np.minimum(H, (np.ceil(bb2[:, 3] / 16.0)).astype(int))
    x1min = np.minimum(xs1, xo1)  # [batch,]
    x1max = np.maximum(xs1, xo1)
    y1min = np.minimum(ys1, yo1)
    y1max = np.maximum(ys1, yo1)
    x2min = np.minimum(xs2, xo2)
    x2max = np.maximum(xs2, xo2)
    y2min = np.minimum(ys2, yo2)
    y2max = np.maximum(ys2, yo2)
    batch = bb1.shape[0]
    feature_vector = np.zeros((batch,features.shape[1],features.shape[2],features.shape[3]))
    for i in range(batch):
        feature_vector[i,:,y1min[i]:y2max[i],x1min[i]:x2max[i]]=features[:,:,y1min[i]:y2max[i],x1min[i]:x2max[i]]
        feature_vector[i,:,y1min[i]:y1max[i],x1min[i]:x1max[i]] = 0
        feature_vector[i,:,y2min[i]:y2max[i],x2min[i]:x2max[i]] = 0
    return feature_vector


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
