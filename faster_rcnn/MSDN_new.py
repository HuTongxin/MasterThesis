#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:35:34 2018

@author: hutong
"""

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import os.path as osp
from torchvision.utils import save_image

from utils.timer import Timer
from utils.HDN_utils import check_relationship_recall
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_hdn import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from RPN import RPN
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
from Spatial_model import get_dual_mask, remove_bg_info, spatial_pyramid_pool
import gmm
from gmm import apply_gmm

import network
from network import Conv2d, FC, Resnet
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16
from MSDN_base import HDN_base
import pdb

DEBUG = False
TIME_IT = cfg.TIME_IT


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep], keep
    return pred_boxes[keep], scores[keep], inds[keep], keep

class Hierarchical_Descriptive_Model(HDN_base):
    def __init__(self,nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                 object_loss_weight, predicate_loss_weight, dropout=False, 
                 use_kmeans_anchors=False, use_kernel=False,
                 disable_spatial_model=False, spatial_type='dual_mask'):
    
        super(Hierarchical_Descriptive_Model, self).__init__(nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                 object_loss_weight, predicate_loss_weight, dropout, use_kmeans_anchors, disable_spatial_model, spatial_type)

        self.rpn = RPN(use_kmeans_anchors)
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        self.roi_pool_rel = RoIPool(7, 7, 1.0/16)
        self.fc6_s = FC(512 * 7 * 7, nhidden, relu=True) 
        self.fc7_s = FC(nhidden, nhidden, relu=False)
        self.fc6_o = FC(512 * 7 * 7, nhidden, relu=True)
        self.fc7_o = FC(nhidden, nhidden, relu=False)
        self.fc6_r = FC(512 * 7 * 7, nhidden, relu=True)   
        self.fc7_r = FC(nhidden, nhidden, relu=False)
               
        # self.conv6 = Resnet(nhidden, nhidden, kernel_size=3)
        # self.conv6_r = Resnet(nhidden, nhidden, kernel_size=3)
        # self.lstm = nn.LSTM(nhidden, nhidden, 2)        
        # self.score_triplet = FC(nhidden, 1, relu=False)
        
        if not self.disable_spatial_model:
            if self.spatial_type == 'remove_bg_info':
                self.conv_union6 = Conv2d(nhidden, nhidden, kernel_size=1)
                self.fc_union7 = FC(25088, 2 * nhidden, relu=True)
                network.weights_normal_init(self.fc_union7, 0.01)  
            if spatial_type == 'dual_mask':
                self.conv6 = Conv2d(2, 96, kernel_size=5)
                self.conv7 = Conv2d(96, 128, kernel_size=5)
                self.conv8 = Conv2d(128, 64, kernel_size=8)
                self.fc9_r = FC(20544, 2 * nhidden, relu=True)
                network.weights_normal_init(self.fc9_r, 0.01)           
            if self.spatial_type == 'gaussian_model':
                self.fc6_union = FC(2448, 2 * nhidden, relu=True)
                network.weights_normal_init(self.fc6_union, 0.01)           
            self.fc10_r= FC(2 * nhidden, nhidden, relu=True)
            network.weights_normal_init(self.fc10_r, 0.01)
        else:
            self.fc8_r = FC(4 * nhidden, nhidden, relu=True)
            network.weights_normal_init(self.fc8_r, 0.01) 
            
        self.fc8_s = FC(4 * nhidden, nhidden, relu=True)
        self.fc8_o = FC(4 * nhidden, nhidden, relu=True)  

        self.score_s = FC(nhidden, self.n_classes_obj, relu=False)
        self.score_r = FC(nhidden, self.n_classes_pred, relu=False)
        self.score_o = FC(nhidden, self.n_classes_obj, relu=False)

        self.boundingbox_s = FC(nhidden, self.n_classes_obj * 4, relu=False)
        self.boundingbox_o = FC(nhidden, self.n_classes_obj * 4, relu=False)
        
        network.weights_normal_init(self.fc8_s, 0.01)
        network.weights_normal_init(self.fc8_o, 0.01)

        network.weights_normal_init(self.score_s, 0.01)
        network.weights_normal_init(self.score_r, 0.01)
        network.weights_normal_init(self.score_o, 0.01)
        # network.weights_normal_init(self.score_triplet, 0.01)
        network.weights_normal_init(self.boundingbox_s, 0.005)   
        network.weights_normal_init(self.boundingbox_o, 0.005)  
        
        self.bad_img_flag=False


    def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None):

        features, object_rois = self.rpn(im_data, im_info, gt_objects)

        if not self.training and gt_objects is not None:
            zeros = np.zeros((gt_objects.shape[0], 1), dtype=gt_objects.dtype)
            object_rois_gt = np.hstack((zeros, gt_objects[:, :4]))
            object_rois_gt = network.np_to_variable(object_rois_gt, is_cuda=True)
            if object_rois.size()[0] >= object_rois_gt.size(0):
                object_rois[:object_rois_gt.size(0)] = object_rois_gt
            else: 
                object_rois_gt = object_rois_gt[:object_rois.size()[0]]
                object_rois = object_rois_gt

        # print 'object_rois.shape', object_rois.size()
        # get object_rois.size(?,5)[0, x1, y1, x2, y2] # ? = 2000,1976,336,165,...

        # print 'features.std'
        # print features.data.std()
        # get feature map W*H*512

        output_proposal_target = \
            self.proposal_target_layer(object_rois, gt_objects, gt_relationships, 
                    self.n_classes_obj, self.voc_sign, self.training)

        if output_proposal_target == 'bad image':
            self.bad_img_flag = True
            return

        roi_data_sub, roi_data_obj, roi_data_rel, mat_phrase, rel_target = output_proposal_target
        # roi_data_object: object_rois(32*5), object_labels(32*1), bbox_targets(32*(151*4)), 
        #                  bbox_inside_weights(32*604), bbox_outside_weights(32*604) 
        # roi_data_predicate: phrase_rois(512*5), phrase_label(512*1)

        sub_rois = roi_data_sub[0]
        obj_rois = roi_data_obj[0]
        rel_rois = roi_data_rel[0]
        # print 'object_rois_num: {}'.format(object_rois.size()[0]) # get object_rois.size(?,5) ?<=256
        # print 'phrase_rois_num: {}'.format(phrase_rois.size()[0]) # get phrase_rois.size(512,5)
        # print 'region_rois_num: {}'.format(region_rois.size()[0])
        
        # roi pool
        pooled_s_features = self.roi_pool(features, sub_rois)  # [batch,512,7,7]
        pooled_o_features = self.roi_pool(features, obj_rois)
        pooled_r_features = self.roi_pool_rel(features, rel_rois)
        # print 'pooled_s_features', pooled_s_features
        # print 'pooled_r_features', pooled_r_features
        # print 'pooled_o_features', pooled_o_features
        
        resize_s_features = pooled_s_features.contiguous().view(pooled_s_features.size()[0],-1)  # [batch,25088]
        resize_o_features = pooled_o_features.contiguous().view(pooled_o_features.size()[0],-1)
        resize_r_features = pooled_r_features.contiguous().view(pooled_r_features.size()[0],-1)

        fc6_s_features = self.fc6_s(resize_s_features)
        if self.dropout:
            fc6_s_features = F.dropout(fc6_s_features, training = self.training)
        fc7_s_features = self.fc7_s(fc6_s_features)
        if self.dropout:
            fc7_s_features = F.dropout(fc7_s_features, training = self.training)
            
        fc6_r_features = self.fc6_r(resize_r_features)
        if self.dropout:
            fc6_r_features = F.dropout(fc6_r_features, training=self.training)
        fc7_r_features = self.fc7_r(fc6_r_features)
        if self.dropout:
            fc7_r_features = F.dropout(fc7_r_features, training=self.training)    
               
        fc6_o_features = self.fc6_o(resize_o_features)
        if self.dropout:
            fc6_o_features = F.dropout(fc6_o_features, training = self.training)
        fc7_o_features = self.fc7_o(fc6_o_features)
        if self.dropout:
            fc7_o_features = F.dropout(fc7_o_features, training = self. training)
        # print 'fc7_s_features', fc7_s_features  # [batch,512]
        # print 'fc7_o_features', fc7_o_features
        
        # bounding box regression
        bbox_s = self.boundingbox_s(F.relu(fc7_s_features))
        bbox_o = self.boundingbox_o(F.relu(fc7_o_features))
        
        embedding = torch.cat((fc7_s_features, fc7_r_features, fc7_o_features),1)  # [batch, 1536] 
        union_r_feature = torch.cat((embedding, fc7_r_features),1)  #[batch, 2048]

        if not self.disable_spatial_model:
            if self.spatial_type == 'remove_bg_info':
                # remember to annotate the previous fc layers
                feature_vector = remove_bg_info(features, sub_rois, obj_rois)  # [batch,512,H,W]
                feature_vector = network.np_to_variable(feature_vector, is_cuda=True)
                feature_vector = self.conv_union6(feature_vector) # [batch,512,H,W]
                batch = feature_vector.size()[0]  
                H = feature_vector.size()[2]
                W = feature_vector.size()[3]
                spp = spatial_pyramid_pool(feature_vector, batch, [H,W], [7]) # [batch, 25088]
                # cat_feature = torch.cat((union_r_feature, spp),1) # [batch, 25136]
                cat_feature = self.fc_union7(spp) #[batch, 1024]
                if self.dropout:
                    cat_feature = F.dropout(cat_feature, training=self.training)
                
            if self.spatial_type == 'dual_mask':                
                mask1 = get_dual_mask(im_info, sub_rois)  # [batch,32,32]
                mask2 = get_dual_mask(im_info, obj_rois)
                mask_pairs = np.zeros((mask1.shape[0], 2, mask1.shape[1], mask1.shape[2]))
                mask_pairs[:, 0, :, :] = mask1
                mask_pairs[:, 1, :, :] = mask2
                mask_pairs = network.np_to_variable(mask_pairs, is_cuda=True)  # [batch,2,32,32]

                mask_pairs = self.conv6(mask_pairs)  # [batch,96,28,28]
                mask_pairs = self.conv7(mask_pairs)  # [batch,128,24,24]
                mask_pairs = self.conv8(mask_pairs)  # [batch,64,17,17]

                spatial_feature = mask_pairs.contiguous().view(mask_pairs.size()[0], -1)  # [batch,18496]
                cat_feature = torch.cat((union_r_feature, spatial_feature), 1)  # [batch,20544]

                cat_feature = self.fc9_r(cat_feature)  #[batch, 1024]
                if self.dropout:
                    cat_feature = F.dropout(cat_feature, training=self.training)

            if self.spatial_type == 'gaussian_model':
                pairs = torch.cat((sub_rois.data[:,1:], obj_rois.data[:,1:]),1) # [batch,8](xs1,ys1,xs2,ys2,xo1,yo1,xo2,yo2)
                pairs = pairs.cpu().numpy()
                spatial_vector = apply_gmm(pairs, r"/home/hutong/tmp/xin/MSDN/tools/", r"gmmmodel_10000randnum_3000pic.m")
                spatial_vector = network.np_to_variable(spatial_vector, is_cuda=True)  # [batch,400]
                
                union_feature = torch.cat((union_r_feature, spatial_vector), 1)  # [batch,2448]
                cat_feature = self.fc6_union(union_feature) #[batch, 1024]
                if self.dropout:
                    cat_feature = F.dropout(cat_feature, training=self.training)
                    
            r_features = self.fc10_r(cat_feature)  #[batch,512]
            if self.dropout:
                r_features = F.dropout(r_features, training=self.training)
                
        else:
            r_features = self.fc8_r(union_r_feature)        

        union_s_feature = torch.cat((embedding, fc7_s_features),1)  # [batch,2048]
        s_features = self.fc8_s(union_s_feature)  #[batch, 512]
        
        union_o_feature = torch.cat((embedding, fc7_o_features),1)
        o_features = self.fc8_o(union_o_feature)
  
       
#        if self.use_resnet: 
#            conv6_s_features = self.conv6(pooled_s_features)   #[batch,512,1,1]
#            conv6_r_features = self.conv6_r(pooled_r_features)
#            conv6_o_features = self.conv6(pooled_o_features)
#        
#            conv6_s_features = conv6_s_features.contiguous().view(conv6_s_features.size()[0],-1)  #[batch,512]
#            conv6_r_features = conv6_r_features.contiguous().view(conv6_r_features.size()[0],-1)
#            conv6_o_features = conv6_o_features.contiguous().view(conv6_o_features.size()[0],-1)
#            # print 'conv6_s_features', conv6_s_features
#            # print 'conv6_r_features', conv6_r_features
#            # print 'conv6_o_features', conv6_o_features
#        
#            union_s_feature = Variable(torch.zeros(conv6_r_features.size()[0], 2, conv6_r_features.size()[1]).cuda())
#            union_s_feature[:,0,:] = conv6_s_features
#            union_s_feature[:,1,:] = conv6_r_features
#            union_o_feature = Variable(torch.zeros(conv6_r_features.size()[0], 2, conv6_r_features.size()[1]).cuda())
#            union_o_feature[:,0,:] = conv6_o_features
#            union_o_feature[:,1,:] = conv6_r_features
#            union_s_feature = union_s_feature.contiguous().view(union_s_feature.size()[0],-1)  #[batch,1024]
#            union_o_feature = union_o_feature.contiguous().view(union_o_feature.size()[0],-1)
#        
#            # bounding box regression before lstm
#            bbox_s = self.boundingbox(F.relu(conv6_s_features))
#            bbox_o = self.boundingbox(F.relu(conv6_o_features))


        cls_score_s = self.score_s(s_features)
        cls_prob_s = F.softmax(cls_score_s)
        cls_score_r = self.score_r(r_features)
        cls_prob_r = F.softmax(cls_score_r)
        cls_score_o = self.score_o(o_features)
        cls_prob_o = F.softmax(cls_score_o)
        # print 'cls_prob_s', cls_prob_s   # [32,151]
        # print 'cls_prob_r', cls_prob_r   # [32,51]
        # print 'cls_prob_o', cls_prob_o   # [32,151]

        if self.training:
            self.cross_entropy_s, self.loss_s_box, self.tp_s, self.tf_s, self.fg_cnt_s, self.bg_cnt_s = \
                          self.build_loss_object(cls_score_s, bbox_s, roi_data_sub)
            self.cross_entropy_o, self.loss_o_box, self.tp_o, self.tf_o, self.fg_cnt_o, self.bg_cnt_o = \
                          self.build_loss_object(cls_score_o, bbox_o, roi_data_obj)
            self.cross_entropy_r, self.tp_r, self.tf_r, self.fg_cnt_r, self.bg_cnt_r = \
                          self.build_loss_cls(cls_score_r, roi_data_rel[1])
            # print 'accuracy: %2.2f%%' % (((self.tp_r + self.tf_r) / float(self.fg_cnt_r + self.bg_cnt_r)) * 100)

        return (cls_prob_s, bbox_s, sub_rois), (cls_prob_r, mat_phrase), (cls_prob_o, bbox_o, obj_rois) 
    

    @staticmethod
    def proposal_target_layer(object_rois, gt_objects, gt_relationships, 
            n_classes_obj, voc_sign, is_training=False):

        """
        ----------
        object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] int
        gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
        gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (-1 for padding)
        # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
        # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        n_classes_obj
        n_classes_pred
        is_training to indicate whether in training scheme
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """

        object_rois = object_rois.data.cpu().numpy()

        output_proposal_target_layer_py = \
            proposal_target_layer_py(object_rois, gt_objects, gt_relationships, 
                n_classes_obj, voc_sign, is_training)
        
        if output_proposal_target_layer_py == 'bad image':
            return 'bad image'
        
        sub_labels, sub_rois, bbox_targets_s, bbox_inside_weights_s, bbox_outside_weights_s, \
            obj_labels, obj_rois, bbox_targets_o, bbox_inside_weights_o, bbox_outside_weights_o, \
            rel_label, rel_rois, mat_phrase, rel_target = output_proposal_target_layer_py

        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        if is_training:
            sub_labels = network.np_to_variable(sub_labels, is_cuda=True, dtype=torch.LongTensor)
            obj_labels = network.np_to_variable(obj_labels, is_cuda=True, dtype=torch.LongTensor)
            bbox_targets_s = network.np_to_variable(bbox_targets_s, is_cuda=True)
            bbox_targets_o = network.np_to_variable(bbox_targets_o, is_cuda=True)
            bbox_inside_weights_s = network.np_to_variable(bbox_inside_weights_s, is_cuda=True)
            bbox_inside_weights_o = network.np_to_variable(bbox_inside_weights_o, is_cuda=True)
            bbox_outside_weights_s = network.np_to_variable(bbox_outside_weights_s, is_cuda=True)
            bbox_outside_weights_o = network.np_to_variable(bbox_outside_weights_o, is_cuda=True)
            rel_label = network.np_to_variable(rel_label, is_cuda=True, dtype=torch.LongTensor)
            rel_target = network.np_to_variable(rel_target, is_cuda=True, dtype=torch.FloatTensor)
            
        sub_rois = network.np_to_variable(sub_rois, is_cuda=True)
        obj_rois = network.np_to_variable(obj_rois, is_cuda=True)
        rel_rois = network.np_to_variable(rel_rois, is_cuda=True)

        return (sub_rois, sub_labels, bbox_targets_s, bbox_inside_weights_s,bbox_outside_weights_s), \
               (obj_rois, obj_labels, bbox_targets_o, bbox_inside_weights_o,bbox_outside_weights_o), \
               (rel_rois, rel_label), mat_phrase, rel_target

    def interpret_HDN(self, cls_prob_s, bbox_s, s_rois, cls_prob_o, bbox_o, o_rois, cls_prob_r, mat_phrase,
                        im_info, nms=True, clip=True, min_score=0.0, top_N=100, use_gt_boxes=False):
        
        scores_s, inds_s = cls_prob_s[:, 1:].data.max(1)
        inds_s += 1
        scores_s, inds_s = scores_s.cpu().numpy(), inds_s.cpu().numpy()
        scores_o, inds_o = cls_prob_o[:, 1:].data.max(1)
        inds_o += 1
        scores_o, inds_o = scores_o.cpu().numpy(), inds_o.cpu().numpy()
        scores_r, inds_r = cls_prob_r[:, 1:].data.max(1)
        inds_r += 1
        scores_r, inds_r = scores_r.cpu().numpy(), inds_r.cpu().numpy()        
        
        keep_s = np.where((inds_s > 0) & (scores_s >= min_score))
        scores_s, inds_s = scores_s[keep_s], inds_s[keep_s]
        keep_o = np.where((inds_o > 0) & (scores_o >= min_score))
        scores_o, inds_o = scores_o[keep_o], inds_o[keep_o]       
        
        # Apply bounding-box regression deltas
        keep_s = keep_s[0]
        box_deltas_s = bbox_s.data.cpu().numpy()[keep_s]
        box_deltas_s = np.asarray([
            box_deltas_s[i, (inds_s[i] * 4): (inds_s[i] * 4 + 4)] for i in range(len(inds_s))
        ], dtype=np.float)
        boxes_s = s_rois.data.cpu().numpy()[keep_s, 1:5] / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes_s = boxes_s
        else:
            pred_boxes_s = bbox_transform_inv_hdn(boxes_s, box_deltas_s)

        if clip:
            pred_boxes_s = clip_boxes(pred_boxes_s, im_info[0][:2] / im_info[0][2])            
        # Apply bounding-box regression deltas
        keep_o = keep_o[0]
        box_deltas_o = bbox_o.data.cpu().numpy()[keep_o]
        box_deltas_o = np.asarray([
            box_deltas_o[i, (inds_o[i] * 4): (inds_o[i] * 4 + 4)] for i in range(len(inds_o))
        ], dtype=np.float)
        boxes_o = o_rois.data.cpu().numpy()[keep_o, 1:5] / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes_o = boxes_o
        else:
            pred_boxes_o = bbox_transform_inv_hdn(boxes_o, box_deltas_o)

        if clip:
            pred_boxes_o = clip_boxes(pred_boxes_o, im_info[0][:2] / im_info[0][2])
        
#        # nms
#        if nms and pred_boxes_s.shape[0] > 0:
#            pred_boxes_s, scores_s, inds_s, keep_keep_s = nms_detections(pred_boxes_s, scores_s, 0.60, inds=inds_s)
#            keep_s = keep_s[keep_keep_s]
#        # nms
#        if nms and pred_boxes_o.shape[0] > 0:
#            pred_boxes_o, scores_o, inds_o, keep_keep_o = nms_detections(pred_boxes_o, scores_o, 0.60, inds=inds_o)
#            keep_o = keep_o[keep_keep_o]
        
        all_scores = scores_r.squeeze() * scores_s.squeeze() * scores_o.squeeze()
        _top_N_list = all_scores.argsort()[::-1][:top_N]
        
        inds_r = inds_r[_top_N_list]
        inds_s = inds_s[_top_N_list]
        inds_o = inds_o[_top_N_list]
        boxes_s = pred_boxes_s[_top_N_list]
        boxes_o = pred_boxes_o[_top_N_list]       

        return inds_r, inds_s, inds_o, boxes_s, boxes_o
               


    def interpret_result(self, cls_prob, bbox_pred, rois, cls_prob_predicate, 
                        mat_phrase, im_info, im_shape, nms=True, clip=True, min_score=0.01, 
                        use_gt_boxes=False):
        scores, inds = cls_prob[:, 0:].data.max(1)
        # inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        predicate_scores, predicate_inds = cls_prob_predicate[:, 0:].data.max(1)
        # predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy(), predicate_inds.cpu().numpy()
        
        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes = boxes
        else:
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)

        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds, keep_keep = nms_detections(pred_boxes, scores, 0.3, inds=inds)
            keep = keep[keep_keep]

        
        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # print 'keep', keep


        for i in range(mat_phrase.shape[0]):
            sub_id = np.where(keep == mat_phrase[i, 0])[0]
            print 's', sub_id
            obj_id = np.where(keep == mat_phrase[i, 1])[0]
            print 'o', obj_id
            if len(sub_id) > 0 and len(obj_id) > 0:
                sub_list = np.append(sub_list, sub_id[0])
                obj_list = np.append(obj_list, obj_id[0])
                pred_list = np.append(pred_list, i)

        predicate_scores = predicate_scores.squeeze()[pred_list]
        final_list = predicate_scores.argsort()[::-1]
        predicate_inds = predicate_inds.squeeze()[pred_list[final_list]]
        sub_list = sub_list[final_list]
        obj_list = obj_list[final_list]
        region_list = mat_phrase[pred_list[final_list], 2:]
        

        return pred_boxes, scores, inds, sub_list, obj_list, predicate_inds, region_list


    def caption(self, im_path, gt_objects=None, gt_regions=None, thr=0.0, nms=False, top_N=100, clip=True, use_beam_search=False):
            image = cv2.imread(im_path)
            # print 'image.shape', image.shape
            im_data, im_scales = self.get_image_blob_noscale(image)
            # print 'im_data.shape', im_data.shape
            # print 'im_scales', im_scales
            if gt_objects is not None:
                gt_objects[:, :4] = gt_objects[:, :4] * im_scales[0]
            if gt_regions is not None:
                gt_regions[:, :4] = gt_regions[:, :4] * im_scales[0]

            im_info = np.array(
                [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
                dtype=np.float32)
            # pdb.set_trace()
            region_result = self(im_data, im_info, gt_objects, gt_regions=gt_regions, use_beam_search=use_beam_search)[2]
            region_caption, bbox_pred, region_rois, logprobs = region_result[:]

            boxes = region_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
            box_deltas = bbox_pred.data.cpu().numpy()
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)
            if clip:
                pred_boxes = clip_boxes(pred_boxes, image.shape)

            # print 'im_scales[0]', im_scales[0]
            return (region_caption.numpy(), logprobs.numpy(), pred_boxes)

    def describe(self, im_path, top_N=10):
            image = cv2.imread(im_path)
            # print 'image.shape', image.shape
            im_data, im_scales = self.get_image_blob_noscale(image)
            # print 'im_data.shape', im_data.shape
            # print 'im_scales', im_scales

            im_info = np.array(
                [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
                dtype=np.float32)

            object_result, predicate_result, region_result = self(im_data, im_info)

            object_boxes, object_scores, object_inds, sub_assignment, obj_assignment, predicate_inds, region_assignment\
                     = self.interpret_result(object_result[0], object_result[1], object_result[2], \
                        predicate_result[0], predicate_result[1], \
                        im_info, image.shape) 

            region_caption, bbox_pred, region_rois, logprobs = region_result[:]
            boxes = region_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
            box_deltas = bbox_pred.data.cpu().numpy()
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, image.shape)

            # print 'im_scales[0]', im_scales[0]
            return (region_caption.numpy(), logprobs.numpy(), pred_boxes, \
                    object_boxes, object_inds, object_scores, \
                sub_assignment, obj_assignment, predicate_inds, region_assignment)


    def evaluate(self, im_data, im_info, gt_objects, gt_relationships, thr=0.5, 
        nms=False, top_Ns = [100], use_gt_boxes=True,  only_predicate=True):
        if use_gt_boxes:
            gt_boxes_object = gt_objects[:, :4]
        else:
            gt_boxes_object = None

        s_result, r_result, o_result = self(im_data, im_info, gt_boxes_object)
        
        cls_prob_s, bbox_s, s_rois = s_result[:3]
        cls_prob_o, bbox_o, o_rois = o_result[:3]
        cls_prob_r, mat_phrase = r_result[:2]

        # interpret the model output
        inds_r, inds_s, inds_o, boxes_s, boxes_o = \
                self.interpret_HDN(cls_prob_s, bbox_s, s_rois, cls_prob_o, bbox_o, o_rois, cls_prob_r, mat_phrase,
                            im_info, nms=nms, top_N=max(top_Ns), use_gt_boxes=use_gt_boxes)
#        np.save('inds_s.npy', inds_s)
#        np.save('inds_r.npy', inds_r)
#        np.save('inds_o.npy', inds_o)
#        np.save('boxes_s.npy', boxes_s)
#        np.save('boxes_o.npy', boxes_o)

        gt_objects[:, :4] /= im_info[0][2]
        rel_cnt, rel_correct_cnt = check_relationship_recall(gt_objects, gt_relationships, 
                                        inds_r, inds_s, inds_o, boxes_s, boxes_o, 
                                        top_Ns, thres=thr, only_predicate=True)

        return rel_cnt, rel_correct_cnt


    def draw_feature(self, im_data,  ef):
        embeded_feature = ef.data.cpu().numpy()
        np.save('embeded_feature.npy', embeded_feature)
        im_data = im_data.cpu().numpy()
        im_data = np.ascontiguousarray(im_data[0].transpose(1, 2, 0))
        im = im_data.copy()
        im[:, :, 0] = ((im_data[:, :, 2] * 0.225 + 0.406) * 255)
        im[:, :, 1] = ((im_data[:, :, 1] * 0.224 + 0.456) * 255)
        im[:, :, 2] = ((im_data[:, :, 0] * 0.229 + 0.485) * 255)
        im = im.astype(np.uint8)
        np.save('original_image.npy', im)
