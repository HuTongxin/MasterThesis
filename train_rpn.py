import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.RPN import RPN # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall, checker

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
import argparse

import pdb


parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=2, help='step to decay the learning rate')

## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy', help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_region', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
args = parser.parse_args()


def main():
    global args
    print "Loading training set and testing set..."
    train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print "Done."

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net = RPN(not args.use_normal_anchors)
    if args.resume_training:
        print 'Resume training from: {}'.format(args.resume_model)
        if len(args.resume_model) == 0:
            raise Exception('[resume_model] not specified')
        network.load_net(args.resume_model, net)
        optimizer = torch.optim.SGD([
                {'params': list(net.parameters())[26:]}, 
                ], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    else:
        print 'Training from scratch...Initializing network...'
        optimizer = torch.optim.SGD(list(net.parameters())[26:], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    network.set_trainable(net.features, requires_grad=False)
    net.cuda()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    best_recall = 0.0
    
    for epoch in range(0, args.max_epoch):
        
        # Training
        # train(train_loader, net, optimizer, epoch)

        # Testing
        recall, RPN_precision, RPN_recall = test(test_loader, net)
        print('Epoch[{epoch:d}]: '
              'Recall: '
              'object: {recall: .3f}%% (Best: {best_recall: .3f}%%)'.format(
               epoch = epoch, recall=recall * 100, best_recall=best_recall * 100))
        print('object: {precision: .3f}%% '
              'object: {recall: .3f}%% '.format(precision=RPN_precision*100, recall=RPN_recall*100))

        # update learning rate
        if epoch % args.step_size == 0:
            args.disable_clip_gradient = True
            args.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

        if np.all(recall > best_recall):
            best_recall = recall
            save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name, epoch))
            network.save_net(save_name, net)


def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    train_loss = network.AverageMeter()
    train_loss_obj_box = network.AverageMeter()
    train_loss_obj_entropy = network.AverageMeter()

    target_net.train()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)

        # Forward pass
        target_net(im_data, im_info.numpy(), gt_objects.numpy()[0])
        # record loss
        loss = target_net.loss
        # total loss
        train_loss.update(loss.data[0], im_data.size(0))
        # object bbox reg
        train_loss_obj_box.update(target_net.loss_box.data[0], im_data.size(0))
        # object score
        train_loss_obj_entropy.update(target_net.cross_entropy.data[0], im_data.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        if not args.disable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  (i + 1) % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch_Time: {batch_time.avg:.3f}s\t'
                  'lr: {lr: f}\t'
                  'Loss: {loss.avg:.4f}\n'
                  '\t[object]: '
                  'cls_loss: {cls_loss_object.avg:.3f}\t'
                  'reg_loss: {reg_loss_object.avg:.3f}\n'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr, 
                   data_time=data_time, loss=train_loss, 
                   cls_loss_object=train_loss_obj_entropy, reg_loss_object=train_loss_obj_box))


def test(test_loader, target_net):
    box_num = 0
    correct_cnt, total_cnt = 0, 0
    precision_correct, precision_total, recall_correct, recall_total = 0, 0, 0, 0
    print '========== Testing ======='
    target_net.eval()

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(test_loader):
        # Forward pass
        object_rois = target_net(im_data, im_info.numpy(), gt_objects.numpy())[1]
        box_num += object_rois.size(0)  # all the objects found by RPN

        # check RPN precision and recall
        # precision = correct objects/ all the objects found   # recall = correct objects/ gt_objects
        precision_correct_t, precision_total_t, recall_correct_t, recall_total_t = checker(object_rois, gt_objects[0].numpy(), 0.3)
        precision_correct += precision_correct_t
        precision_total += precision_total_t
        recall_correct += recall_correct_t
        recall_total += recall_total_t

        correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_objects[0].numpy(), 50)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 and i > 0:
            print('[{0}/{6}]  Time: {1:2.3f}s/img).'
                  '\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:d}/{5:d})'.format(
                i + 1, batch_time.avg,
                box_num / float(i + 1), correct_cnt / float(total_cnt) * 100, correct_cnt, total_cnt,
                len(test_loader)))

    recall = correct_cnt / float(total_cnt)
    RPN_precision = precision_correct / float(precision_total)
    RPN_recall = recall_correct / float(recall_total)
    print '====== Done Testing ===='
    return recall, RPN_precision, RPN_recall


if __name__ == '__main__':
    main()
