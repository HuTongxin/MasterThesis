import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils.HDN_utils import check_relationship_recall
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_target_layer_hdn import proposal_target_layer as proposal_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_object
from fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from RPN import RPN
from fast_rcnn.config import cfg
from faster_rcnn.additional_model.Spatial_model import GaussianMixtureModel, DualMask, GeometricSpatialFeature
from faster_rcnn.additional_model.Iterative_Structure import BrnnStructure, Concat, TranslationEmbedding, GraphicalModel

import network
from network import FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool, MaskRoIPool, DualMaskRoIPool
from MSDN_base import HDN_base

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
                 disable_spatial_model=False, spatial_type='dual_mask',
                 pool_type='roi_pooling',
                 disable_iteration_model=False, iteration_type='cat_embed',
                 idx2obj=None, idx2rel=None):
    
        super(Hierarchical_Descriptive_Model, self).__init__(nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                                object_loss_weight, predicate_loss_weight, dropout, use_kmeans_anchors,
                                disable_spatial_model, spatial_type, pool_type, disable_iteration_model, iteration_type)

        self.rpn = RPN(use_kmeans_anchors)
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        if self.pool_type == 'roi_pooling':
            self.roi_pool_rel = RoIPool(7, 7, 1.0/16)
        if self.pool_type == 'spatial_attention':
            self.mask_roi_pool = MaskRoIPool(7, 7, 1.0/16)
        if self.pool_type == 'dual_roipooling':
            self.dualmask_roi_pool = DualMaskRoIPool(7, 7, 1.0/16)
        self.fc6 = FC(512 * 7 * 7, nhidden, relu=True)
        self.fc7 = FC(nhidden, nhidden, relu=True)
        self.fc6_r = FC(512 * 7 * 7, nhidden, relu=True)   
        self.fc7_r = FC(nhidden, nhidden, relu=True)
        
        if not self.disable_spatial_model:
            if spatial_type == 'dual_mask':
                self.dm = DualMask(nhidden)
            if self.spatial_type == 'gaussian_model':
                self.gmm = GaussianMixtureModel(25488, nhidden)
            self.fc10_r= FC(2 * nhidden, nhidden, relu=True)
            network.weights_normal_init(self.fc10_r, 0.01)
        else:
            self.gsf = GeometricSpatialFeature(nhidden, dropout)

        TransEmbedding = False
        if TransEmbedding:
            self.TransE = TranslationEmbedding(nhidden, dropout)

        if not self.disable_iteration_model:
            if self.iteration_type == 'use_brnn':
                self.lstm = BrnnStructure(nhidden, dropout)
            if self.iteration_type == 'cat_embed':
                self.embed = Concat(nhidden, dropout)
            if self.iteration_type == 'iteration':
                self.iter = GraphicalModel(nhidden, dropout)
        else:
            self.fc8 = FC(2 * nhidden, nhidden, relu=True)
            self.fc9 = FC(nhidden, nhidden, relu=True)
            network.weights_normal_init(self.fc8, 0.01)
            network.weights_normal_init(self.fc9, 0.01)

        self.score = FC(nhidden, self.n_classes_obj, relu=False)
        self.score_r = FC(nhidden, self.n_classes_pred, relu=False)

        self.boundingbox = FC(nhidden, self.n_classes_obj * 4, relu=False)

        network.weights_normal_init(self.score, 0.01)
        network.weights_normal_init(self.score_r, 0.01)
        network.weights_normal_init(self.boundingbox, 0.005)
        
        self.bad_img_flag = False

        # for plotting of training
        self.idx2obj = idx2obj
        self.idx2rel = idx2rel
        self.trainImgCount = 0

    def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None):

        features, object_rois, rpn_scores = self.rpn(im_data, im_info, gt_objects)  # features:[1,512,H,W]

        proposals, after_nms_scores, _ = nms_detections(object_rois[:, 1:].data.cpu().numpy(), rpn_scores.data.cpu().numpy(), 0.30)
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        object_rois = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        object_rois = network.np_to_variable(object_rois, is_cuda=True, dtype=torch.LongTensor)

        if not self.training and gt_objects is not None:
            # remove the gt rois with heights or weights which are smaller than 1
            object_rois = self.remove_bad_roi(gt_objects)

        output_proposal_target = \
            self.proposal_target_layer(object_rois, gt_objects, gt_relationships, 
                    self.n_classes_obj, self.voc_sign, self.training)

        if output_proposal_target == 'bad image':
            self.bad_img_flag = True
            return

        roi_data_sub, roi_data_obj, roi_data_rel, mat_phrase, rel_target, roi_data_object = output_proposal_target

        sub_rois = roi_data_sub[0]
        obj_rois = roi_data_obj[0]  # get object_rois.size(?,5) ?<=256
        rel_rois = roi_data_rel[0]

        # roi pool
        pooled_s_features = self.roi_pool(features, sub_rois)  # [batch,512,7,7]
        pooled_o_features = self.roi_pool(features, obj_rois)
        if self.pool_type == 'spatial_attention':
            pooled_r_features = self.mask_roi_pool(features, sub_rois, obj_rois)
        if self.pool_type == 'roi_pooling':
            pooled_r_features = self.roi_pool_rel(features, rel_rois)
        if self.pool_type == 'dual_roipooling':
            pooled_r_features = self.dualmask_roi_pool(features, sub_rois, obj_rois)

        resize_s_features = pooled_s_features.contiguous().view(pooled_s_features.size()[0], -1)  # [batch,25088]
        resize_o_features = pooled_o_features.contiguous().view(pooled_o_features.size()[0], -1)
        resize_r_features = pooled_r_features.contiguous().view(pooled_r_features.size()[0], -1)

        fc6_s_features = self.fc6(resize_s_features)
        fc6_o_features = self.fc6(resize_o_features)
        if self.dropout:
            fc6_s_features = F.dropout(fc6_s_features, training=self.training)
            fc6_o_features = F.dropout(fc6_o_features, training=self.training)

        fc7_s_features = self.fc7(fc6_s_features)  # [batch,512]
        fc7_o_features = self.fc7(fc6_o_features)
        if self.dropout:
            fc7_s_features = F.dropout(fc7_s_features, training=self.training)
            fc7_o_features = F.dropout(fc7_o_features, training=self.training)
        
        # bounding box regression
        bbox_s = self.boundingbox(F.relu(fc7_s_features))
        bbox_o = self.boundingbox(F.relu(fc7_o_features))

        if not self.disable_spatial_model:

            if self.spatial_type == 'dual_mask':
                cat_feature = self.dm(im_info, sub_rois, obj_rois, resize_r_features, self.dropout, self.training)

            if self.spatial_type == 'gaussian_model':
                cat_feature = self.gmm(sub_rois, obj_rois, resize_r_features, self.dropout, self.training)
                    
            r_features = self.fc10_r(cat_feature)  # [batch,512]
            if self.dropout:
                r_features = F.dropout(r_features, training=self.training)
                
        else:
            fc6_r_features = self.fc6_r(resize_r_features)
            if self.dropout:
                fc6_r_features = F.dropout(fc6_r_features, training=self.training)
            r_features = self.fc7_r(fc6_r_features)
            if self.dropout:
                r_features = F.dropout(r_features, training=self.training)

            fc7_s_features, fc7_o_features, r_features = self.gsf(sub_rois, obj_rois, fc7_s_features, fc7_o_features, r_features)

        TransEmbedding = False
        if TransEmbedding:
            r_features = self.TransE(fc7_s_features, fc7_o_features, r_features)

        if not self.disable_iteration_model:
            if self.iteration_type == 'use_brnn':
                s_features, r_features, o_features = self.lstm(fc7_s_features, r_features, fc7_o_features)
            if self.iteration_type == 'cat_embed':
                s_features, r_features, o_features = self.embed(fc7_s_features, r_features, fc7_o_features)
            if self.iteration_type == 'iteration':
                s_features, r_features, o_features = self.iter(fc7_s_features, r_features, fc7_o_features)
        else:
            union_s_feature = torch.cat((fc7_s_features, r_features), 1)  # [batch,1024]
            fc8_s_features = self.fc8(union_s_feature)  # [batch, 512]
            s_features = self.fc9(fc8_s_features)

            union_o_feature = torch.cat((fc7_o_features, r_features), 1)
            fc8_o_features = self.fc8(union_o_feature)
            o_features = self.fc9(fc8_o_features)

        cls_score_s = self.score(s_features)
        cls_prob_s = F.softmax(cls_score_s)  # [32,151]
        cls_score_r = self.score_r(r_features)
        cls_prob_r = F.softmax(cls_score_r)  # [32,51]
        cls_score_o = self.score(o_features)
        cls_prob_o = F.softmax(cls_score_o)  # [32,151]

        # ------------------------
        # build total loss
        # ------------------------
        if self.training:
            self.cross_entropy_s, self.loss_s_box, self.tp_s, self.tf_s, self.fg_cnt_s, self.bg_cnt_s = \
                          self.build_loss_object(cls_score_s, bbox_s, roi_data_sub)
            self.cross_entropy_o, self.loss_o_box, self.tp_o, self.tf_o, self.fg_cnt_o, self.bg_cnt_o = \
                          self.build_loss_object(cls_score_o, bbox_o, roi_data_obj)
            self.cross_entropy_r, self.tp_r, self.tf_r, self.fg_cnt_r, self.bg_cnt_r = \
                          self.build_loss_cls(cls_score_r, roi_data_rel[1])
        # ------------------------
        # end
        # ------------------------

        s_result = (cls_prob_s, bbox_s, sub_rois)
        r_result = (cls_prob_r, mat_phrase)
        o_result = (cls_prob_o, bbox_o, obj_rois)

        # ------------------------
        # plot of training
        # ------------------------
        plot_picture = False
        if plot_picture:
            img_info_tuple = (im_data, im_info, self.idx2obj, self.idx2rel)
            rpn_output = (proposals, after_nms_scores)
            net_output = (s_result, r_result, o_result)
            ground_truth = (gt_objects, gt_relationships)
            proposal_target = (roi_data_sub, roi_data_obj, roi_data_rel)
            self.plot_train(img_info_tuple, rpn_output, net_output, proposal_target, ground_truth)
        # ------------------------
        # end
        # ------------------------

        return s_result, r_result, o_result

    @staticmethod
    def remove_bad_roi(gt_objects):
        """remove the gt rois with heights or weights which are smaller than 1"""
        check_gt = gt_objects[:, :4]
        width_gt = (check_gt[:, 2] - check_gt[:, 0])
        height_gt = (check_gt[:, 3] - check_gt[:, 1])
        gt_objects_without_error = gt_objects[np.where((width_gt > 1) * (height_gt > 1))]

        zeros = np.zeros((gt_objects_without_error.shape[0], 1), dtype=gt_objects.dtype)
        object_rois_gt = np.hstack((zeros, gt_objects_without_error[:, :4]))
        object_rois_gt = network.np_to_variable(object_rois_gt, is_cuda=True)
        object_rois = object_rois_gt
        return object_rois

    def plot_train(self, img_info, rpn_output, net_output, proposal_target, ground_truth):
        print('train checker for image {}----------------------------------------------'.format(self.trainImgCount))
        im2show_original, im2show_rpn_output, im2show_output, im2show_proposal, im2show_real = \
            self.train_checker(img_info, rpn_output, net_output, proposal_target, ground_truth)
        if self.trainImgCount < 40:
            cv2.imwrite('demoImg/train/img_{}_original.jpg'.format(self.trainImgCount), im2show_original)
            cv2.imwrite('demoImg/train/img_{}_rpn_output.jpg'.format(self.trainImgCount), im2show_rpn_output)
            cv2.imwrite('demoImg/train/img_{}_output.jpg'.format(self.trainImgCount), im2show_output)
            cv2.imwrite('demoImg/train/img_{}_proposal.jpg'.format(self.trainImgCount), im2show_proposal)
            cv2.imwrite('demoImg/train/img_{}_real.jpg'.format(self.trainImgCount), im2show_real)
            self.trainImgCount = self.trainImgCount + 1

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

        # add an additional block for object classification
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_object(object_rois, gt_objects, None, n_classes_obj, is_training)

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

            # add an additional block for object classification
            labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
            bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
            bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
            bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)

        sub_rois = network.np_to_variable(sub_rois, is_cuda=True)
        obj_rois = network.np_to_variable(obj_rois, is_cuda=True)
        rel_rois = network.np_to_variable(rel_rois, is_cuda=True)
        rois = network.np_to_variable(rois, is_cuda=True)

        return (sub_rois, sub_labels, bbox_targets_s, bbox_inside_weights_s,bbox_outside_weights_s), \
               (obj_rois, obj_labels, bbox_targets_o, bbox_inside_weights_o,bbox_outside_weights_o), \
               (rel_rois, rel_label), mat_phrase, rel_target, \
               (rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights)

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
        boxes_s = s_rois.data.cpu().numpy()[keep_s, 1:5]  # / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes_s = boxes_s
        else:
            pred_boxes_s = bbox_transform_inv_hdn(boxes_s, box_deltas_s)

        if clip:
            pred_boxes_s = clip_boxes(pred_boxes_s, im_info[0][:2]) # / im_info[0][2])
        # Apply bounding-box regression deltas
        keep_o = keep_o[0]
        box_deltas_o = bbox_o.data.cpu().numpy()[keep_o]
        box_deltas_o = np.asarray([
            box_deltas_o[i, (inds_o[i] * 4): (inds_o[i] * 4 + 4)] for i in range(len(inds_o))
        ], dtype=np.float)
        boxes_o = o_rois.data.cpu().numpy()[keep_o, 1:5]  #  / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes_o = boxes_o
        else:
            pred_boxes_o = bbox_transform_inv_hdn(boxes_o, box_deltas_o)

        if clip:
            pred_boxes_o = clip_boxes(pred_boxes_o, im_info[0][:2])  # / im_info[0][2])
        
        all_scores = scores_r.squeeze() * scores_s.squeeze() * scores_o.squeeze()
        _top_N_list = all_scores.argsort()[::-1][:top_N]
        
        inds_r = inds_r[_top_N_list]
        inds_s = inds_s[_top_N_list]
        inds_o = inds_o[_top_N_list]
        boxes_s = pred_boxes_s[_top_N_list]
        boxes_o = pred_boxes_o[_top_N_list]       

        return inds_r, inds_s, inds_o, boxes_s, boxes_o

    def interpret(self, cls_prob_s, bbox_s, s_rois, cls_prob_o, bbox_o, o_rois, cls_prob_r,
                  im_info, clip=True, min_score=0.0, top_N=100, use_gt_boxes=False):

        scores_s, inds_s = cls_prob_s[:, 1:].data.max(1)
        inds_s += 1
        scores_s, inds_s = scores_s.cpu().numpy(), inds_s.cpu().numpy()
        scores_o, inds_o = cls_prob_o[:, 1:].data.max(1)
        inds_o += 1
        scores_o, inds_o = scores_o.cpu().numpy(), inds_o.cpu().numpy()

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
        boxes_s = s_rois.data.cpu().numpy()[keep_s, 1:5]
        if use_gt_boxes:
            clip = False
            pred_boxes_s = boxes_s
        else:
            pred_boxes_s = bbox_transform_inv_hdn(boxes_s, box_deltas_s)

        if clip:
            pred_boxes_s = clip_boxes(pred_boxes_s, im_info[0][:2])

        # Apply bounding-box regression deltas
        keep_o = keep_o[0]
        box_deltas_o = bbox_o.data.cpu().numpy()[keep_o]
        box_deltas_o = np.asarray([
            box_deltas_o[i, (inds_o[i] * 4): (inds_o[i] * 4 + 4)] for i in range(len(inds_o))
        ], dtype=np.float)
        boxes_o = o_rois.data.cpu().numpy()[keep_o, 1:5]
        if use_gt_boxes:
            clip = False
            pred_boxes_o = boxes_o
        else:
            pred_boxes_o = bbox_transform_inv_hdn(boxes_o, box_deltas_o)

        if clip:
            pred_boxes_o = clip_boxes(pred_boxes_o, im_info[0][:2])

        cls_prob_r = cls_prob_r[:, 1:].data.cpu().numpy()
        a = cls_prob_r.reshape(1, -1)[0]
        order = a.argsort()[::-1]
        cls_r = []
        for idx, top_N in enumerate(top_N):
            cls = cls_prob_r.copy()
            cls[cls < a[order[top_N-1]]] = 0
            x = np.where(cls != 0)[0]
            y = np.where(cls != 0)[1]
            cls[x, y] = y + 1
            cls_r.append(cls)

        return cls_r, inds_s, inds_o, pred_boxes_s, pred_boxes_o

    def evaluate(self, im_data, im_info, gt_objects, gt_relationships, thr=0.5,
        nms=False, top_Ns = [100], use_gt_boxes=True,  union_overlap=True):
        if use_gt_boxes:
            gt_boxes_object = gt_objects[:, :4]
        else:
            gt_boxes_object = None

        s_result, r_result, o_result = self(im_data, im_info, gt_boxes_object)
        # s_result, r_result, o_result, bicls_result = self(im_data, im_info, gt_boxes_object)  # bicls_result[batch,1]

        cls_prob_s, bbox_s, s_rois = s_result  # cls_prob_s[batch,151] bbox_s[batch,604] s_rois[batch,5]
        cls_prob_o, bbox_o, o_rois = o_result
        cls_prob_r, mat_phrase = r_result  # cls_prob_r[batch,51]

        cls_r, inds_s, inds_o, boxes_s, boxes_o = \
                self.interpret(cls_prob_s, bbox_s, s_rois, cls_prob_o, bbox_o, o_rois, cls_prob_r,
                  im_info, top_N=top_Ns, use_gt_boxes=use_gt_boxes)

        rel_cnt, rel_correct_cnt = check_relationship_recall(gt_objects, gt_relationships,
                                        cls_r, inds_s, inds_o, boxes_s, boxes_o,
                                        top_Ns, use_gt_boxes=use_gt_boxes, thres=thr, union_overlap=union_overlap)

        return rel_cnt, rel_correct_cnt

    def train_checker(self, img_info, rpn_output, net_output, proposal_target, ground_truth, N=10):
        """plot outputs, its targets and the dataset ground truth of training

        Args:
            img_info: (im_data, im_info, idx2obj, idx2rel)
            rpn_output: (object_rois, after_nms_scores)
            net_output: (s_result, r_result, o_result)
            proposal_target: (roi_data_sub, roi_data_obj, roi_data_rel)
            ground_truth: (gt_objects, gt_relationships)
        """
        import cv2

        def box_union(box1, box2):
            return np.concatenate((np.minimum(box1[:, :2], box2[:, :2]), np.maximum(box1[:, 2:], box2[:, 2:])), 1)

        def rebuild_image(im_data):
            im_data = np.ascontiguousarray(im_data.transpose(1, 2, 0))
            im = im_data.copy()
            im[:, :, 2] = ((im_data[:, :, 2] * 0.225 + 0.406) * 255)
            im[:, :, 1] = ((im_data[:, :, 1] * 0.224 + 0.456) * 255)
            im[:, :, 0] = ((im_data[:, :, 0] * 0.229 + 0.485) * 255)
            im = im.astype(np.uint8)
            return im

        im_data, im_info, idx2obj, idx2rel = img_info
        proposals, after_nms_scores = rpn_output
        s_result, r_result, o_result = net_output
        roi_data_sub, roi_data_obj, roi_data_rel = proposal_target
        gt_objects, gt_relationships = ground_truth

        # draw the original image
        im_data = im_data[0].numpy()
        im2show_original = rebuild_image(im_data)

        # draw the after nms rpn output
        im2show = im2show_original.copy()

        rpn_output = np.concatenate((proposals, after_nms_scores.reshape(after_nms_scores.shape[0], 1)), 1)
        for i in rpn_output:
            cv2.rectangle(im2show, (int(i[0]), int(i[1])),
                          (int(i[2]), int(i[3])), (0, 255, 0), 2)
            cv2.putText(im2show, str(i[4]),
                        (int(i[0]), int(i[1] + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), thickness=1)

        im2show_rpn_output = im2show.copy()

        # draw the network output
        im2show = im2show_original.copy()

        cls_prob_s, bbox_s, s_rois = s_result[:3]
        cls_prob_o, bbox_o, o_rois = o_result[:3]
        cls_prob_r, mat_phrase = r_result[:2]

        inds_r, inds_s, inds_o, boxes_s, boxes_o = \
            self.interpret_HDN(cls_prob_s, bbox_s, s_rois, cls_prob_o,
                               bbox_o, o_rois, cls_prob_r, mat_phrase, im_info)

        output_s = np.concatenate((boxes_s, inds_s.reshape(inds_s.shape[0], 1)), 1)
        output_o = np.concatenate((boxes_o, inds_o.reshape(inds_o.shape[0], 1)), 1)
        boxes_r = box_union(boxes_s, boxes_o)
        output_r = np.concatenate((boxes_r, inds_r.reshape(inds_r.shape[0], 1)), 1)

        for s in output_s:
            cv2.rectangle(im2show, (int(s[0]), int(s[1])),
                          (int(s[2]), int(s[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(s[4])],
                        (int(s[0]), int(s[1] + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), thickness=1)
        for o in output_o:
            cv2.rectangle(im2show, (int(o[0]), int(o[1])),
                          (int(o[2]), int(o[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(o[4])],
                        (int(o[0]), int(o[1] + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), thickness=1)
        for r in output_r:
            cv2.rectangle(im2show, (int(r[0]), int(r[1])),
                          (int(r[2]), int(r[3])), (138, 43, 226), 1)
            cv2.putText(im2show, idx2rel[int(r[4])],
                        (int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (138, 43, 226), thickness=1)

        im2show_output = im2show.copy()

        # draw the proposal target
        def make_mat(cls_vector, size):
            cls_mat = Variable(torch.zeros(size)).cuda()
            cls_vector = cls_vector.squeeze()
            for idx, item in enumerate(cls_vector):
                cls_mat[idx, item.data.cpu().numpy()[0]] = 1
            return cls_mat

        im2show = im2show_original.copy()

        s_rois, cls_s, bbox_s = roi_data_sub[:3]
        cls_s = make_mat(cls_s, cls_prob_s.size())
        o_rois, cls_o, bbox_o = roi_data_obj[:3]
        cls_o = make_mat(cls_o, cls_prob_o.size())
        cls_r = roi_data_rel[1]
        # print('train checker cls_r: {}'.format(cls_r.squeeze().cpu().data.numpy()))
        cls_r = make_mat(cls_r, cls_prob_r.size())

        inds_r, inds_s, inds_o, boxes_s, boxes_o = \
            self.interpret_HDN(cls_s, bbox_s, s_rois, cls_o,
                               bbox_o, o_rois, cls_r, mat_phrase, im_info)
        # print('train checker inds_r: {}'.format(inds_r))

        output_s = np.concatenate((boxes_s, inds_s.reshape(inds_s.shape[0], 1)), 1)
        output_o = np.concatenate((boxes_o, inds_o.reshape(inds_o.shape[0], 1)), 1)
        boxes_r = box_union(boxes_s, boxes_o)
        output_r = np.concatenate((boxes_r, inds_r.reshape(inds_r.shape[0], 1)), 1)

        for s in output_s:
            cv2.rectangle(im2show, (int(s[0]), int(s[1])),
                          (int(s[2]), int(s[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(s[4])],
                        (int(s[0]), int(s[1] + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), thickness=1)
        for o in output_o:
            cv2.rectangle(im2show, (int(o[0]), int(o[1])),
                          (int(o[2]), int(o[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(o[4])],
                        (int(o[0]), int(o[1] + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), thickness=1)
        for r in output_r:
            cv2.rectangle(im2show, (int(r[0]), int(r[1])),
                          (int(r[2]), int(r[3])), (138, 43, 226), 1)
            cv2.putText(im2show, idx2rel[int(r[4])],
                        (int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (138, 43, 226), thickness=1)

        im2show_proposal = im2show.copy()

        # draw the ground truth of objects
        im2show = im2show_original.copy()
        if not gt_objects is None:
            for gt_object in gt_objects:
                cv2.rectangle(im2show, tuple(gt_object[0:2]),
                              tuple(gt_object[2:4]), (0, 0, 255), 2)
                cv2.putText(im2show, idx2obj[int(gt_object[4])],
                            (int(gt_object[0]), int(gt_object[1] + 15)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), thickness=1)

            # draw the gt_relationships
            ind = np.where(gt_relationships > 0)
            box1 = gt_objects[ind[0]][:, :4]
            box2 = gt_objects[ind[1]][:, :4]
            union_box = box_union(box1, box2)
            label_relationship = gt_relationships[ind[0], ind[1]].reshape(len(ind[0]), 1)
            gt_relationship = np.concatenate((union_box, label_relationship), 1)
            for gt in gt_relationship:
                cv2.rectangle(im2show, (int(gt[0]), int(gt[1])),
                              (int(gt[2]), int(gt[3])), (255, 69, 0), 1)
                cv2.putText(im2show, idx2rel[int(gt[4])],
                            (int((gt[0] + gt[2]) / 2), int((gt[1] + gt[3]) / 2)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 69, 0), thickness=1)

        # print('train_checker gt_rel: {}'.format(label_relationship.squeeze()))
        im2show_real = im2show.copy()

        return im2show_original, im2show_rpn_output, im2show_output, im2show_proposal, im2show_real

    def detect(self, im_data, im_info, idx2obj, idx2rel, gt_objects, gt_relationships, N=5):
        """Detect the objects and their relationships 
        in a given image and save the result. 
        If ground truth is given, draw it out, too.
        
        Args:
            im_data: normalized image data from data loader
            im_info: image information from data loader
            idx2obj: convert object index to object word. 
                This list is in dataset._object_classes
            idx2rel: convert relationship index to relationship word.
                This list is in dataset._predicate_classes
            gt_objects: ground truth bboxs and their classes
            gt_relationships : ground truth relationships
        """      
        import cv2
        
        def box_union(box1, box2):    
            return np.concatenate((np.minimum(box1[:, :2], box2[:, :2]), np.maximum(box1[:, 2:], box2[:, 2:])), 1)
        
        def rebuild_image(im_data):
            im_data = np.ascontiguousarray(im_data.transpose(1,2,0))
            im = im_data.copy()
            im[:,:,2]=((im_data[:,:,2]*0.225+0.406)*255)
            im[:,:,1]=((im_data[:,:,1]*0.224+0.456)*255)
            im[:,:,0]=((im_data[:,:,0]*0.229+0.485)*255)
            im = im.astype(np.uint8)
            return im

        # draw the original image
        im = im_data[0].numpy()
        im2show_original = rebuild_image(im)

        # draw the network output
        im2show = im2show_original.copy()
        gt_boxes_object = gt_objects[:, :4] # * im_info[2]
        s_result, r_result, o_result = self(im_data, im_info, gt_boxes_object)
        
        cls_prob_s, bbox_s, s_rois = s_result[:3]
        cls_prob_o, bbox_o, o_rois = o_result[:3]
        cls_prob_r, mat_phrase = r_result[:2]

        inds_r, inds_s, inds_o, boxes_s, boxes_o = \
                self.interpret_HDN(cls_prob_s, bbox_s, s_rois, cls_prob_o, 
                                   bbox_o, o_rois, cls_prob_r, mat_phrase, im_info)
  
        output_s = np.concatenate((boxes_s[:N,:], (inds_s[:N]).reshape(N,1)), 1)
        output_o = np.concatenate((boxes_o[:N,:], (inds_o[:N]).reshape(N,1)), 1)
        boxes_r = box_union(boxes_s[:N,:], boxes_o[:N,:])
        output_r = np.concatenate((boxes_r, (inds_r[:N]).reshape(N,1)), 1)

        for s in output_s:
            cv2.rectangle(im2show, (int(s[0]), int(s[1])), 
                              (int(s[2]), int(s[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(s[4])], 
                            (int(s[0]), int(s[1] + 15)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            1, (0, 255, 0), thickness=1)
        for o in output_o:
            cv2.rectangle(im2show, (int(o[0]), int(o[1])), 
                              (int(o[2]), int(o[3])), (0, 255, 0), 2)
            cv2.putText(im2show, idx2obj[int(o[4])], 
                            (int(o[0]), int(o[1] + 15)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            1, (0, 255, 0), thickness=1)
        for r in output_r:
            cv2.rectangle(im2show, (int(r[0]), int(r[1])), 
                              (int(r[2]), int(r[3])), (138, 43, 226), 1)
            cv2.putText(im2show, idx2rel[int(r[4])],
                        (int((r[0]+r[2])/2), int((r[1]+r[3])/2)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (138, 43, 226), thickness=1)

        im2show_output = im2show.copy()

        # draw the ground truth of objects
        im2show = im2show_original.copy()
        if not gt_objects is None:
            for gt_object in gt_objects:
                cv2.rectangle(im2show, tuple(gt_object[0:2]), 
                              tuple(gt_object[2:4]), (0, 0, 255), 2)
                cv2.putText(im2show, idx2obj[int(gt_object[4])],
                            (int(gt_object[0]), int(gt_object[1] + 15)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), thickness=1)
                
            # draw the gt_relationships
            ind = np.where(gt_relationships>0)
            box1 = gt_objects[ind[0]][:,:4]
            box2 = gt_objects[ind[1]][:,:4]
            union_box = box_union(box1, box2)
            label_relationship = gt_relationships[ind[0], ind[1]].reshape(len(ind[0]),1)
            gt_relationship = np.concatenate((union_box, label_relationship),1)
            for gt in gt_relationship:
                cv2.rectangle(im2show, (int(gt[0]), int(gt[1])),
                              (int(gt[2]), int(gt[3])), (255, 69, 0), 1)
                cv2.putText(im2show, idx2rel[int(gt[4])],
                            (int((gt[0]+gt[2])/2), int((gt[1]+gt[3])/2)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 69, 0), thickness=1)

        im2show_real = im2show.copy()
            
        return im2show_original, im2show_output, im2show_real
