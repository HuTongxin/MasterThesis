# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import pdb

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


#  object_rois, object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mat_object, \
#              phrase_rois, phrase_label, mat_phrase, region_rois, region_seq, mat_region = \
#              proposal_target_layer_py(object_rois, region_rois, gt_objects, gt_relationships,
#                  gt_regions, n_classes_obj, n_classes_pred, is_training)




def proposal_target_layer(object_rois, gt_objects, gt_relationships, 
                n_classes_obj, voc_eos, is_training):

    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] float
    #     gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
    #     gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (imdb.eos for padding)
    #     # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
    #     # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    #     n_classes_obj
    #     n_classes_pred
    #     is_training to indicate whether in training scheme

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois

    # assert is_training == True, 'Evaluation Code haven\'t been implemented'

    # if there are any gt rois whose heights or weights are smaller than 1, print all gt
    check_gt = gt_objects[:, :4]
    width_gt = (check_gt[:, 2] - check_gt[:, 0])
    height_gt = (check_gt[:, 3] - check_gt[:, 1])
    if ((width_gt < 1) + (height_gt < 1)).sum():
        # print('gt roi {}'.format(check_gt))
        pass

    # Sample rois with classification labels and bounding box regression
    # targets
    if is_training:
        all_rois = object_rois

        # get rid of the gt rois with heights or weights which are smaller than 1
        gt_objects_without_error = gt_objects[np.where((width_gt > 1) * (height_gt > 1))]

        zeros = np.zeros((gt_objects_without_error.shape[0], 1), dtype=gt_objects.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_objects_without_error[:, :4])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        output_sample_rois = _sample_rois(all_rois, gt_objects, \
                    gt_relationships, 1, n_classes_obj, voc_eos, is_training)

        if output_sample_rois == 'bad image':
            return 'bad image'

        sub_labels, sub_rois, bbox_targets_s, bbox_inside_weights_s, \
           obj_labels, obj_rois, bbox_targets_o, bbox_inside_weights_o, \
           rel_labels, rel_rois, mat_phrase, rel_target = output_sample_rois


        # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
        sub_labels = sub_labels.reshape(-1, 1)   # add the labels for subjects
        obj_labels = obj_labels.reshape(-1, 1)   # add the labels for objects
        bbox_targets_s = bbox_targets_s.reshape(-1, n_classes_obj * 4)
        bbox_targets_o = bbox_targets_o.reshape(-1, n_classes_obj * 4)
        bbox_inside_weights_s = bbox_inside_weights_s.reshape(-1, n_classes_obj * 4)
        bbox_inside_weights_o = bbox_inside_weights_o.reshape(-1, n_classes_obj * 4)
        bbox_outside_weights_s = np.array(bbox_inside_weights_s > 0).astype(np.float32)
        bbox_outside_weights_o = np.array(bbox_inside_weights_o > 0).astype(np.float32)
        rel_labels = rel_labels.reshape(-1, 1)
        rel_target = rel_target.reshape(-1, 1)
        
    else:
        sub_rois, obj_rois, rel_rois, mat_phrase = _setup_connection(object_rois)
        sub_labels, bbox_targets_s, bbox_inside_weights_s, bbox_outside_weights_s, \
        obj_labels, bbox_targets_o, bbox_inside_weights_o, bbox_outside_weights_o, \
        rel_labels, rel_target = [None] * 10
    # print 'region_roi', region_roi
    # print 'object_rois'
    # print object_rois
    # print 'phrase_rois'
    # print phrase_rois
    # get object_rois.size(1280,5) phrase_rois.size(2560,5)

    if DEBUG:
        # print 'region_roi'
        # print region_roi
        # print 'object num fg: {}'.format((object_labels > 0).sum())
        # print 'object num bg: {}'.format((object_labels == 0).sum())
        # print 'relationship num fg: {}'.format((phrase_labels > 0).sum())
        # print 'relationship num bg: {}'.format((phrase_labels == 0).sum())
        count_sub = 1
        fg_num_s = (sub_labels > 0).sum()
        bg_num_s = (sub_labels == 0).sum()
        print 'sub num fg avg: {}'.format(fg_num_s / count_sub)
        print 'sub num bg avg: {}'.format(bg_num_s / count_sub)
        print 'ratio: {:.3f}'.format(float(fg_num_s) / float(bg_num_s))
        count_rel = 1
        fg_num_rel = (rel_labels > 0).sum()
        bg_num_rel = (rel_labels == 0).sum()
        print 'relationship num fg avg: {}'.format(fg_num_rel / count_rel)
        print 'relationship num bg avg: {}'.format(bg_num_rel / count_rel)
        print 'ratio: {:.3f}'.format(float(fg_num_rel) / float(bg_num_rel))
        count_obj = 1
        fg_num_o = (obj_labels > 0).sum()
        bg_num_o = (obj_labels == 0).sum()
        print 'obj num fg avg: {}'.format(fg_num_o / count_obj)
        print 'obj num bg avg: {}'.format(bg_num_o / count_obj)
        print 'ratio: {:.3f}'.format(float(fg_num_o) / float(bg_num_o)) 
        # print mat_object.shape
        # print mat_phrase.shape
        # print 'region_roi'
        # print region_roi

    # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
    # mps_phrase [phrase_batchsize, 2 + n_region]
    # mps_region [region_batchsize, n_phrase]
    assert sub_rois.shape[1] == 5
    assert obj_rois.shape[1] == 5
    assert rel_rois.shape[1] == 5

    return sub_labels, sub_rois, bbox_targets_s, bbox_inside_weights_s, bbox_outside_weights_s, \
            obj_labels, obj_rois, bbox_targets_o, bbox_inside_weights_o, bbox_outside_weights_o, \
            rel_labels, rel_rois, mat_phrase, rel_target


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #     # Optionally normalize targets by a precomputed mean and stdev
    #     targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    #                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(object_rois, gt_objects, gt_relationships, num_images, num_classes, voc_eos, is_training):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images   # 256
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)    # 0.25*256

    overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))
    #  overlaps: [object_rois[0],gt_objects[0]]
    gt_assignment = overlaps.argmax(axis=1)  # for each roi, find the best fitting gt
    max_overlaps = overlaps.max(axis=1)  # get the overlap rate for each roi
    labels = gt_objects[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))   #<=64
    # fg_rois_per_this_image = int(min(bg_inds.size, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    # if fg_inds.size>=64, select random 64 from fg_inds; if fg_inds<64 select all fg_inds

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image   #>=192
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # bg_rois_per_this_image = fg_rois_per_this_image
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # maybe more than 192, maybe smaller than 192

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = object_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_objects[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

#### prepare relationships targets


    rel_per_image = int(cfg.TRAIN.BATCH_SIZE_RELATIONSHIP / num_images)   # 512
    rel_bg_num = rel_per_image
    if fg_inds.size > 0:
        assert fg_inds.size == fg_inds.shape[0]
        id_i, id_j = np.meshgrid(xrange(fg_inds.size), xrange(fg_inds.size), indexing='ij') # Grouping the input object rois
        id_i = id_i.reshape(-1)
        id_j = id_j.reshape(-1)
        # id_i: array([ 0,  0,  0, ..., 63, 63, 63])  id_j: array([ 0,  1,  2, ..., 61, 62, 63])
        pair_labels = gt_relationships[gt_assignment[fg_inds[id_i]], gt_assignment[fg_inds[id_j]]]
        fg_id_rel = np.where(pair_labels > 0)[0]
        rel_fg_num = fg_id_rel.size
        rel_fg_num = int(min(np.round(rel_per_image * cfg.TRAIN.FG_FRACTION_RELATIONSHIP), rel_fg_num))  #<=256
        # print 'rel_fg_num'
        # print rel_fg_num
        if rel_fg_num > 0:
            fg_id_rel = npr.choice(fg_id_rel, size=rel_fg_num, replace=False)
        else:
            fg_id_rel = np.empty(0, dtype=int)
        rel_labels_fg = pair_labels[fg_id_rel]
        sub_assignment_fg = id_i[fg_id_rel]
        obj_assignment_fg = id_j[fg_id_rel]
        sub_list_fg = fg_inds[sub_assignment_fg]
        obj_list_fg = fg_inds[obj_assignment_fg]
        rel_bg_num = rel_per_image - rel_fg_num   # rel_bg_num = int(min(1, rel_per_image * cfg.TRAIN.FG_FRACTION_RELATIONSHIP - rel_fg_num))

    phrase_labels = np.zeros(rel_bg_num, dtype=np.float)
    sub_assignment = npr.choice(xrange(keep_inds.size), size=rel_bg_num, replace=True)
    obj_assignment = npr.choice(xrange(keep_inds.size), size=rel_bg_num, replace=True)
    sub_list = keep_inds[sub_assignment]
    obj_list = keep_inds[obj_assignment]

    if fg_inds.size > 0:
        rel_labels = np.append(phrase_labels, rel_labels_fg)
        # to do the binary classification
        target_num = rel_labels_fg.size
        targets = np.ones(target_num, dtype=np.float)
        rel_target = np.append(phrase_labels, targets)

        sub_list = np.append(sub_list, sub_list_fg)
        obj_list = np.append(obj_list, obj_list_fg)
        sub_assignment = np.append(sub_assignment, sub_assignment_fg)
        obj_assignment = np.append(obj_assignment, obj_assignment_fg)

    sub_labels = gt_objects[gt_assignment[sub_list], 4]
    obj_labels = gt_objects[gt_assignment[obj_list], 4]
    sub_rois = object_rois[sub_list, :]
    obj_rois = object_rois[obj_list, :]
    rel_rois = box_union(object_rois[sub_list, :], object_rois[obj_list, :])

    ### prepare connection matrix
    mat_object, mat_phrase = _prepare_mat(sub_assignment, obj_assignment, keep_inds.size)

    bbox_target_data_s = _compute_targets(sub_rois[:, 1:5], gt_objects[gt_assignment[sub_list], :4], sub_labels)
    bbox_targets_s, bbox_inside_weights_s = _get_bbox_regression_labels(bbox_target_data_s, num_classes)
    bbox_target_data_o = _compute_targets(obj_rois[:, 1:5], gt_objects[gt_assignment[obj_list], :4], obj_labels)
    bbox_targets_o, bbox_inside_weights_o = _get_bbox_regression_labels(bbox_target_data_o, num_classes)

    return sub_labels, sub_rois, bbox_targets_s, bbox_inside_weights_s, \
               obj_labels, obj_rois, bbox_targets_o, bbox_inside_weights_o, \
               rel_labels, rel_rois, mat_phrase, rel_target


def _setup_connection(object_rois):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    roi_num = cfg.TEST.BBOX_NUM  # 64
    keep_inds = np.array(range(min(roi_num, object_rois.shape[0])))
    # keep_inds = np.array(range(object_rois.shape[0]))  # <=32

    id_i, id_j = _generate_pairs(keep_inds) # Grouping the input object rois and remove the diagonal items # 32*31 = 992
    
    sub_rois = object_rois[id_i, :]
    obj_rois = object_rois[id_j, :]
    rel_rois = box_union(object_rois[id_i, :], object_rois[id_j, :])  # 992

    _, mat_phrase = _prepare_mat(id_i, id_j, rel_rois.shape[0])
#    
#    rel_id = np.array(range(rel_rois.shape[0]))
#    rel_num = rel_rois.shape[0]
#    test_num = 64
#    test_num = int(min(rel_num, test_num))
#    rel_id = npr.choice(rel_id, size=test_num, replace=False)
#    
#    sub_list = id_i[rel_id]
#    obj_list = id_j[rel_id]
#    mat_phrase = np.zeros((sub_list.size, 2), dtype=np.int64)
#    mat_phrase[:, 0] = sub_list
#    mat_phrase[:, 1] = obj_list
#       
#    rel_rois = rel_rois[rel_id,:]
#    sub_rois = sub_rois[rel_id,:]
#    obj_rois = obj_rois[rel_id,:]
    
    # print 'before union', object_rois[id_i[0], :], object_rois[id_j[0], :]
    # print 'after union', phrase_rois[0, :]

    return sub_rois, obj_rois, rel_rois, mat_phrase

def box_union(box1, box2):    
    return np.concatenate((np.minimum(box1[:, :3], box2[:, :3]), np.maximum(box1[:, 3:], box2[:, 3:])), 1)

def _prepare_mat(sub_list, obj_list, object_batchsize):
    # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
    # mps_phrase [phrase_batchsize, 2 + n_region]
    # mps_region [region_batchsize, n_phrase]

    
    phrase_batchsize = sub_list.size
    # print 'phrase_batchsize', phrase_batchsize

    mat_object = np.zeros((object_batchsize, 2, phrase_batchsize), dtype=np.int64)
    mat_phrase = np.zeros((phrase_batchsize, 2), dtype=np.int64)
    mat_phrase[:, 0] = sub_list
    mat_phrase[:, 1] = obj_list

    for i in xrange(phrase_batchsize):
        mat_object[sub_list[i], 0, i] = 1
        mat_object[obj_list[i], 1, i] = 1

    return mat_object, mat_phrase

def _generate_pairs(ids):
    id_i, id_j = np.meshgrid(ids, ids, indexing='ij') # Grouping the input object rois
    id_i = id_i.reshape(-1) 
    id_j = id_j.reshape(-1)
    # remove the diagonal items
    id_num = len(ids)
    diagonal_items = np.array(range(id_num))
    diagonal_items = diagonal_items * id_num + diagonal_items
    all_id = range(len(id_i))
    selected_id = np.setdiff1d(all_id, diagonal_items)
    id_i = id_i[selected_id]
    id_j = id_j[selected_id]

    return id_i, id_j
