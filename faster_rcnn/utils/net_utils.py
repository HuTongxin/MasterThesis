import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from cython_bbox import bbox_overlaps, bbox_intersections

def get_model_name(arguments):


    if arguments.nesterov:
        arguments.model_name += '_nesterov'

    if arguments.use_kernel_function:
        arguments.model_name += '_with_kernel'
    if arguments.load_RPN or arguments.resume_training:
        arguments.model_name += '_alt'
    else:
        arguments.model_name += '_end2end'
    if arguments.dropout:
        arguments.model_name += '_dropout'
    arguments.model_name += '_{}'.format(arguments.dataset_option)

    if arguments.resume_training:
        arguments.model_name += '_resume'

    if arguments.optimizer == 0:
        arguments.model_name += '_SGD'
        arguments.solver = 'SGD'
    elif arguments.optimizer == 1:
        arguments.model_name += '_Adam'
        arguments.solver = 'Adam'
    elif arguments.optimizer == 2:    
        arguments.model_name += '_Adagrad'
        arguments.solver = 'Adagrad'
    else:
        raise Exception('Unrecognized optimization algorithm specified!')

    return arguments

def group_features(net_):
    vgg_features_fix = list(net_.rpn.features.parameters())[:8]
    vgg_features_var = list(net_.rpn.features.parameters())[8:]
    vgg_feature_len = len(list(net_.rpn.features.parameters()))
    rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len
    rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]
    hdn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):]
    print 'vgg feature length:', vgg_feature_len
    print 'rpn feature length:', rpn_feature_len
    print 'HDN feature length:', len(hdn_features)
    return vgg_features_fix, vgg_features_var, rpn_features, hdn_features


def check_recall(rois, gt_objects, top_N, thres=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois.cpu().data.numpy()[:top_N, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:4], dtype=np.float))

    overlap_gt = np.amax(overlaps, axis=0)
    correct_cnt = np.sum(overlap_gt >= thres)
    total_cnt = overlap_gt.size 
    return correct_cnt, total_cnt


def checker(rois, gt_objects, thres=0.7):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois.cpu().data.numpy()[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))

    max_overlaps = np.amax(overlaps, axis=1)
    precision_correct = np.sum(max_overlaps >= thres)
    precision_total = max_overlaps.size

    overlaps_gt = np.amax(overlaps, axis=0)
    recall_correct = np.sum(overlaps_gt >= thres)
    recall_total = overlaps_gt.size
    return precision_correct, precision_total, recall_correct, recall_total


def check_relationship_recall(gt_objects, gt_relationships,
        cls_r, inds_s, inds_o, boxes_s, boxes_o,
        top_Ns, thres=0.5, use_gt_boxes=True, union_overlap=True):

    def box_union(box1, box2):
        return np.concatenate((np.minimum(box1[:, :2], box2[:, :2]), np.maximum(box1[:, 2:], box2[:, 2:])), 1)

    boxes_union = box_union(boxes_s, boxes_o)
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5]
    gt_obj = gt_objects[gt_rel_obj_idx, :5]
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

    gt_union = box_union(gt_sub, gt_obj)

    rel_cnt = len(gt_rel)
    rel_correct_cnt = np.zeros(len(top_Ns))
    
    sub_overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes_s[:, :4], dtype=np.float),
        np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes_o[:, :4], dtype=np.float),
        np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))
    union_overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes_union[:, :4], dtype=np.float),
        np.ascontiguousarray(gt_union[:, :4], dtype=np.float))

    for idx, top_N in enumerate(top_Ns):

        if use_gt_boxes:
            for gt_id in xrange(rel_cnt):
                fg_candidate = np.where(np.logical_and(
                   sub_overlaps[:, gt_id] == 1,
                   obj_overlaps[:, gt_id] == 1))[0]

                for candidate_id in fg_candidate:
                    for cls_id in range(cls_r[idx].shape[1]):
                        if cls_r[idx][candidate_id, cls_id] == gt_rel[gt_id]:
                            rel_correct_cnt[idx] += 1
                            break

        elif union_overlap:
            for gt_id in xrange(rel_cnt):
                flag = 0
                fg_candidate = np.where(union_overlaps[:, gt_id] >= thres)[0]

                for candidate_id in fg_candidate:
                    if flag == 1:
                        break

                    for cls_id in range(cls_r[idx].shape[1]):
                        if cls_r[idx][candidate_id, cls_id] == gt_rel[gt_id] and \
                             inds_s[candidate_id] == gt_sub[gt_id, 4] and \
                             inds_o[candidate_id] == gt_obj[gt_id, 4]:
                            rel_correct_cnt[idx] += 1
                            flag = 1
                            break

        else:
            for gt_id in xrange(rel_cnt):
                fg_candidate = np.where(np.logical_and(
                   sub_overlaps[:, gt_id] >= thres,
                   obj_overlaps[:, gt_id] >= thres))[0]

                for candidate_id in fg_candidate:

                    for cls_id in range(cls_r[idx].shape[1]):
                        if cls_r[idx][candidate_id, cls_id] == gt_rel[gt_id] and \
                                inds_s[candidate_id] == gt_sub[gt_id, 4] and \
                                inds_o[candidate_id] == gt_obj[gt_id, 4]:
                            rel_correct_cnt[idx] += 1
                            break

    return rel_cnt, rel_correct_cnt


def relationship_checker(gt_objects, gt_relationships, bicls, boxes_s, boxes_o, thres=0.99):
    '''
    :param gt_objects: (gt_num, 5) [x1,y1,x2,y2,cls]
    :param gt_relationships: (gt_num, gt_num)
    :param bicls: prediction of 'have relationship or not' (gt*(gt-1), 1)
    :param boxes_s: (gt*(gt-1), 5) [0,x1,y1,x2,y2]
    :param boxes_o: (gt*(gt-1), 5)
    :return:
    '''
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0)  # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5]
    gt_obj = gt_objects[gt_rel_obj_idx, :5]
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

    recall_total = len(gt_rel)
    precision_total = np.sum(bicls >= 0.5)
    recall_correct = 0

    sub_overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes_s[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes_o[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))

    for gt_id in xrange(recall_total):
        fg_candidate = np.where(np.logical_and(
            sub_overlaps[:, gt_id] == 1,
            obj_overlaps[:, gt_id] == 1))[0]

        for candidate_id in fg_candidate:
            if bicls[candidate_id] >= 0.5:
                recall_correct += 1
                break

    precision_correct = recall_correct

    return precision_correct, precision_total, recall_correct, recall_total