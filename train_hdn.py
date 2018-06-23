import os
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse
import json


import torch
import cv2

from faster_rcnn import network
from faster_rcnn.MSDN import Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.HDN_utils import get_model_name, group_features

import pdb

# To log the training process
from tensorboard_logger import configure, log_value

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
TIME_IT = cfg.TIME_IT

parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

# Training parameters
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=20, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, default = 2, help='Step size for reduce learning rate')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--load_RPN', action='store_true', help='To end-to-end train from the scratch')
parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')


# structure settings
parser.add_argument('--feature_len', type=int, default=512, help='The expected feature length of message passing')
parser.add_argument('--dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--use_kernel_function', action='store_true')
parser.add_argument('--disable_spatial_model', action='store_true', help='To disable the Spatial Model')
parser.add_argument('--spatial_type', type=str, default='dual_mask', help='Select the Spatial Model[dual_mask | gaussian_model | remove_bg_info]')
parser.add_argument('--pool_type', type=str, default='roi_pooling', help='Select the Pooling Type[roi_pooling | spatial_attention | dual_roipooling]')
parser.add_argument('--disable_iteration_model', action='store_true', help='To disable the Iteration Model')
parser.add_argument('--iteration_type', type=str, default='cat_embed', help='Select the Iteration Type[use_brnn | cat_embed | iteration]')
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--saved_model_path', type=str, default = 'model/pretrained_models/VGG_imagenet.npy', help='The Model used for initialize')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/Test45', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='HDN', help='The name for saving model.')
parser.add_argument('--nesterov', action='store_true', help='Set to use the nesterov for SGD')
parser.add_argument('--optimizer', type=int, default=0, help='which optimizer used for optimize language model [0: SGD | 1: Adam | 2: Adagrad]')

parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()


optimizer_select = 0


def main():
    global args, optimizer_select
    # To set the model name automatically
    print args
    lr = args.lr
    args = get_model_name(args)
    print 'Model name: {}'.format(args.model_name)

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 1234)

    print("Loading training set and testing set..."),
    train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Model declaration
    net = Hierarchical_Descriptive_Model(nhidden=args.feature_len,
                 n_object_cats=train_set.num_object_classes, 
                 n_predicate_cats=train_set.num_predicate_classes, 
                 n_vocab=train_set.voc_size,
                 voc_sign=train_set.voc_sign,
                 object_loss_weight=train_set.inverse_weight_object, 
                 predicate_loss_weight=train_set.inverse_weight_predicate,
                 dropout=args.dropout, 
                 use_kmeans_anchors=not args.use_normal_anchors, 
                 use_kernel = args.use_kernel_function,
                 disable_spatial_model = args.disable_spatial_model,
                 spatial_type = args.spatial_type,
                 pool_type = args.pool_type,
                 disable_iteration_model = args.disable_iteration_model,
                 iteration_type = args.iteration_type,
                 idx2obj=train_set._object_classes,
                 idx2rel=train_set._predicate_classes)

    params = list(net.parameters())
    for param in params:
        print param.size()
    print net 

    # To group up the features
    
    vgg_features_fix, vgg_features_var, rpn_features, hdn_features = group_features(net)

    # Setting the state of the training model
    net.cuda()
    net.train()
    logger_path = "log/logger45/{}".format(args.model_name)
    if os.path.exists(logger_path):
        shutil.rmtree(logger_path)
    configure(logger_path, flush_secs=5) # setting up the logger

    network.set_trainable(net, True)
    network.set_trainable(net.rpn, False)
    #  network.weights_normal_init(net, dev=0.01)
    if args.load_RPN:
        print 'Loading pretrained RPN: {}'.format(args.saved_model_path)
        args.train_all = False
        network.load_net(args.saved_model_path, net.rpn)
        net.reinitialize_fc_layers()
        optimizer_select = 1       

    elif args.resume_training:
        print 'Resume training from: {}'.format(args.resume_model)
        if len(args.resume_model) == 0:
            raise Exception('[resume_model] not specified')
        network.load_net(args.resume_model, net)
        args.train_all = True
        optimizer_select = 2

    else:
        print 'Training from scratch.'
        net.rpn.initialize_parameters()
        net.reinitialize_fc_layers()
        optimizer_select = 0
        args.train_all = True

    optimizer = network.get_optimizer(lr,optimizer_select, args, 
                vgg_features_var, rpn_features, hdn_features)

    target_net = net
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    top_Ns = [50, 100]
    best_recall = np.zeros(len(top_Ns))

    if args.evaluate:
        # show_result(test_loader, test_set, net, 5)
        recall = test(test_loader, net, top_Ns)
        print('======= Testing Result =======') 
        for idx, top_N in enumerate(top_Ns):
            print('[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
                top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100))
        print('==============================')
    else:
        for epoch in range(0, args.max_epoch):           
            # Training
            train_loss_list = train(train_loader, target_net, optimizer, epoch)
            print(train_loss_list)
            save_loss_list_name = os.path.join(args.output_dir, 'train_loss_epoch_{}.json'.format(epoch))
            with open(save_loss_list_name, 'w') as fout:
                json.dump(train_loss_list, fout)

            # snapshot the state
            save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))

            recall = test(test_loader, net, top_Ns)

            if np.all(recall > best_recall):
                best_recall = recall
                save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name))
                network.save_net(save_name, net)
                print('\nsave model: {}'.format(save_name))

            print('Epoch[{epoch:d}]:'.format(epoch = epoch)),
            for idx, top_N in enumerate(top_Ns):
                print('\t[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
                    top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100)),

            # updating learning policy
            if epoch % args.step_size == 0 and epoch > 0:
                lr /= 10
                args.lr = lr
                print '[learning rate: {}]'.format(lr)
            
                args.enable_clip_gradient = False
                args.train_all = True
                optimizer_select = 2

                # update optimizer and corresponding requires_grad state
                optimizer = network.get_optimizer(lr, optimizer_select, args, 
                            vgg_features_var, rpn_features, hdn_features)


def train(train_loader, target_net, optimizer, epoch):
    global args
    # Overall loss logger
    global overall_train_loss
    global overall_train_rpn_loss
    loss_list = list()

    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    # Total loss
    train_loss = network.AverageMeter()
    # object related loss
    train_s_cls_loss = network.AverageMeter()
    train_o_cls_loss = network.AverageMeter()
    train_s_box_loss = network.AverageMeter()
    train_o_box_loss = network.AverageMeter()
    # relationship cls loss
    train_r_cls_loss = network.AverageMeter()

    # RPN loss
    train_rpn_loss = network.AverageMeter()
    # object
    accuracy_s = network.AccuracyMeter()
    accuracy_o = network.AccuracyMeter()
    accuracy_r = network.AccuracyMeter()

    target_net.train()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)
        target_net(im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0])

        if target_net.bad_img_flag:
            target_net.bad_img_flag = False
            continue

        # Determine the loss function
        if args.train_all:
            loss = target_net.loss + target_net.rpn.loss
        else:
            loss = target_net.loss

        train_loss.update(target_net.loss.data.cpu().numpy()[0], im_data.size(0))
        train_s_cls_loss.update(target_net.cross_entropy_s.data.cpu().numpy()[0], im_data.size(0))
        train_o_cls_loss.update(target_net.cross_entropy_o.data.cpu().numpy()[0], im_data.size(0))
        train_s_box_loss.update(target_net.loss_s_box.data.cpu().numpy()[0], im_data.size(0))
        train_o_box_loss.update(target_net.loss_o_box.data.cpu().numpy()[0], im_data.size(0))
        train_r_cls_loss.update(target_net.cross_entropy_r.data.cpu().numpy()[0], im_data.size(0))
        
        train_rpn_loss.update(target_net.rpn.loss.data.cpu().numpy()[0], im_data.size(0))
        overall_train_loss.update(target_net.loss.data.cpu().numpy()[0], im_data.size(0))
        overall_train_rpn_loss.update(target_net.rpn.loss.data.cpu().numpy()[0], im_data.size(0))
        
        accuracy_s.update(target_net.tp_s, target_net.tf_s, target_net.fg_cnt_s, target_net.bg_cnt_s)
        accuracy_o.update(target_net.tp_o, target_net.tf_o, target_net.fg_cnt_o, target_net.bg_cnt_o)
        accuracy_r.update(target_net.tp_r, target_net.tf_r, target_net.fg_cnt_r, target_net.bg_cnt_r)

        optimizer.zero_grad()
        loss.backward()
        if args.enable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging the training loss
        if  (i + 1) % args.log_interval == 0:
            loss_list.append(train_loss.avg)
            print('\nEpoch: [{0}][{1}/{2}] [lr: {lr}] [Solver: {solver}]\n'
                  '\tBatch_Time: {batch_time.avg: .3f}s\t'
                  'FRCNN Loss: {loss.avg: .4f}\t'
                  'RPN Loss: {rpn_loss.avg: .4f}'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr, 
                   loss=train_loss, rpn_loss=train_rpn_loss, solver=args.solver))

            print('\t[Loss]\ts_cls_loss: %.4f\ts_box_loss: %.4f' %
                  (train_s_cls_loss.avg, train_s_box_loss.avg)),
            print('\tr_cls_loss: %.4f,' % (train_r_cls_loss.avg)),
            print('\t[Loss]\to_cls_loss: %.4f\to_box_loss: %.4f' %
                  (train_o_cls_loss.avg, train_o_box_loss.avg)),
            print('\n\t[s]\ttp: %.2f, \tfg=%d' %
                  (accuracy_s.ture_pos*100., accuracy_s.foreground))
            print('\t[r]\ttp: %.2f, \ttf: %.2f, \tfg/bg=(%d/%d)' %
                  (accuracy_r.ture_pos*100., accuracy_r.true_neg*100., accuracy_r.foreground, accuracy_r.background))
            print('\t[o]\ttp: %.2f, \tfg=%d' %
                  (accuracy_o.ture_pos * 100., accuracy_o.foreground))

            # logging to tensor board
            log_value('FRCNN loss', overall_train_loss.avg, overall_train_loss.count)
            log_value('RPN_loss loss', overall_train_rpn_loss.avg, overall_train_rpn_loss.count)

    return loss_list


def test(test_loader, net, top_Ns):
    """test the network, print the number of correct predictions
    
    Args:
        test_loader: torch.utils.data.DataLoader(train_set)
        net: Hierarchical_Descriptive_Model()
        top_Ns: [50, 100]
    Res:
        Recall: rel_cnt_correct / rel_cnt
    """

    global args

    print '========== Testing ======='
    net.eval()
    
    rel_cnt = 0.
    rel_cnt_correct = np.zeros(len(top_Ns))
    precision_correct, precision_total, recall_correct, recall_total = 0, 0, 0, 0

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(test_loader):       
        # Forward pass
        total_cnt_t, rel_cnt_correct_t = net.evaluate(
            im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0], top_Ns=top_Ns, nms=True)

        rel_cnt += total_cnt_t
        rel_cnt_correct += rel_cnt_correct_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 500 == 0 and i > 0:
            for idx, top_N in enumerate(top_Ns):
                print '[%d/%d][Evaluation] Top-%d Recall: %2.3f%%' % (
                    i+1, len(test_loader), top_N, rel_cnt_correct[idx] / float(rel_cnt) * 100)

    recall = rel_cnt_correct / rel_cnt
    print '====== Done Testing ===='

    return recall


def show_result(test_loader, test_set, net, N):
    '''to print the test image
    input: image
    outout: gt_objects(bounding box and class)
            gt_relationships(class)
            test_objects(bounding box and class)
            test_relationships(class)
    '''
    net.eval()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(test_loader):
        im2show_original, im2show_output, im2show_real = \
            net.detect(im_data, im_info, test_set._object_classes, test_set._predicate_classes,
                       gt_objects.numpy()[0], gt_relationships.numpy()[0], N)

        cv2.imwrite('demoImg/test/img_{}_original.jpg'.format(i), im2show_original)
        cv2.imwrite('demoImg/test/img_{}_output.jpg'.format(i), im2show_output)
        cv2.imwrite('demoImg/test/img_{}_real.jpg'.format(i), im2show_real)
        if i == 50:
            print('images saved')
            break
    return


if __name__ == '__main__':
    torch.backends.cudnn.enabled=True
    main()
