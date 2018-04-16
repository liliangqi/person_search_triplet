# -----------------------------------------------------
# Train Spatial Invariant Person Search Network
#
# Author: Liangqi Li
# Creating Date: Mar 31, 2018
# Latest rectified: Apr 16, 2018
# -----------------------------------------------------
import os
import argparse

import torch
import yaml
from torch.autograd import Variable
import time
import random

from __init__ import clock_non_return
from dataset import PersonSearchDataset
from model import SIPN
from losses import triplet_loss


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--net', default='res50', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--pre_model', default='', type=str)

    args = parser.parse_args()

    return args


def cuda_mode(args):
    """set cuda"""
    if torch.cuda.is_available() and '-1' not in args.gpu_ids:
        cuda = True
        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    else:
        cuda = False

    return cuda


def train_model(dataset, net, lr, optimizer, num_epochs, use_cuda, save_dir):
    """Train the model"""

    all_epoch_loss = 0
    df = dataset.train_all
    start = time.time()
    net.train()

    if use_cuda:
        net.cuda()

    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    current_iter = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        if epoch in [2, 4]:
            lr *= config['gamma']  # TODO: use lr_scheduel
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config['gamma']

        # TODO: get num_pid from dataset rather than from the net
        num_pid = net.num_pid
        for pid in range(num_pid):
            im_names = set(df[df['pid'] == pid]['imname'])
            query_name = list(im_names)[random.randint(0, len(im_names))]
            # TODO: contain flipped query image into galleries
            gallery_names = list(im_names - set(query_name))

            # TODO: Maybe we can compute loss after processing all g images
            # Create empty dicts list to save features or losses of galleries
            # galleries_info = []

            q_im, q_roi, q_im_info = dataset.get_query_im(query_name, pid)
            q_im = q_im.transpose([0, 3, 1, 2])

            if use_cuda:
                q_im = Variable(torch.from_numpy(q_im).cuda())
                q_roi = Variable(torch.from_numpy(q_roi).float().cuda())
            else:
                q_im = Variable(torch.from_numpy(q_im))
                q_roi = Variable(torch.from_numpy(q_roi).float())

            q_feat = net(q_im, q_roi, q_im_info, model='query')

            flip = [True] * len(gallery_names) + [False] * len(gallery_names)
            for g_name, flipped in zip(gallery_names * 2, flip):
                im, gt_boxes, im_info = dataset.get_gallery_im(g_name, flipped)
                im = im.transpose([0, 3, 1, 2])
                current_iter += 1

                if use_cuda:
                    im = Variable(torch.from_numpy(im).cuda())
                    gt_boxes = Variable(
                        torch.from_numpy(gt_boxes).float().cuda())
                else:
                    im = Variable(torch.from_numpy(im))
                    gt_boxes = Variable(torch.from_numpy(gt_boxes).float())

                # `det_loss` is a tuple of four losses of detection
                det_loss, pid_label, reid_feat = net(im, gt_boxes, im_info)

                # Note that -1 in `pid_label` refers to unlabeled identities
                label_mask = (pid_label != num_pid).data
                pid_label_drop = pid_label[label_mask]

                # label_mask = label_mask.expand(pid_label.size(0),
                #                                reid_feat.size(1))
                # reid_feat_drop = reid_feat[label_mask].view(
                #     pid_label_drop.size(0), reid_feat.size(1))

                # A better way to implement the mask
                reid_feat_drop = reid_feat[label_mask.nonzero().squeeze()]

                # Forward propagation
                reid_loss = triplet_loss(q_feat, pid, reid_feat_drop,
                                         pid_label_drop, mode='hard')

                # Backward propagation
                optimizer.zero_grad()
                total_loss = det_loss[0] + det_loss[1] + det_loss[2] + \
                    det_loss[3] + reid_loss
                total_loss.backward()
                optimizer.step()

                all_epoch_loss += total_loss.data[0]
                average_loss = all_epoch_loss / current_iter

            end = time.time()
            print('Epoch {:2d}, person {:4d}/{:4d}, average loss: {:.6f}, lr: '
                  '{:.2e}'.format(epoch+1, pid+1, num_pid, average_loss, lr))
            print('>>>> rpn_cls: {:.6f}'.format(det_loss[0].data[0]))
            print('>>>> rpn_box: {:.6f}'.format(det_loss[1].data[0]))
            print('>>>> cls: {:.6f}'.format(det_loss[2].data[0]))
            print('>>>> box: {:.6f}'.format(det_loss[3].data[0]))
            print('>>>> reid: {:.6f}'.format(reid_loss.data[0]))
            print('time cost: {:.3f}s/person'.format(
                (end - start) / (pid + 1)))

        # for step in range(len(dataset)):
        #     im, gt_boxes, im_info = dataset.next()
        #     im = im.transpose([0, 3, 1, 2])
        #
        #     if use_cuda:
        #         im = Variable(torch.from_numpy(im).cuda())
        #         gt_boxes = Variable(torch.from_numpy(gt_boxes).float().cuda()
        # )
        #     else:
        #         im = Variable(torch.from_numpy(im))
        #         gt_boxes = Variable(torch.from_numpy(gt_boxes).float())
        #
        #     losses = net(im, gt_boxes, im_info)
        #     optimizer.zero_grad()
        #     total_loss = sum(losses)
        #     total_loss.backward()
        #     optimizer.step()
        #
        #     all_epoch_loss += total_loss.data[0]
        #     current_iter = epoch * len(dataset) + step + 1
        #     average_loss = all_epoch_loss / current_iter
        #
        #     if (step+1) % config['disp_interval'] == 0:
        #         end = time.time()
        #         print('Epoch {:2d}, iter {:5d}, average loss: {:.6f}, lr: '
        #               '{:.2e}'.format(epoch+1, step+1, average_loss, lr))
        #         print('>>>> rpn_cls: {:.6f}'.format(losses[0].data[0]))
        #         print('>>>> rpn_box: {:.6f}'.format(losses[1].data[0]))
        #         print('>>>> cls: {:.6f}'.format(losses[2].data[0]))
        #         print('>>>> box: {:.6f}'.format(losses[3].data[0]))
        #         print('>>>> reid: {:.6f}'.format(losses[4].data[0]))
        #         print('time cost: {:.3f}s/iter'.format(
        #             (end - start) / (epoch * len(dataset) + (step + 1))))

        epoch_end = time.time()
        print('\nEntire epoch time cost: {:.2f} hours\n'.format(
            (epoch_end - epoch_start) / 3600))

        # Save the trained model after each epoch
        save_name = os.path.join(save_dir, 'sipn_{}.pth'.format(epoch + 1))
        torch.save(net.state_dict(), save_name)


@clock_non_return
def main():

    opt = parse_args()
    use_cuda = cuda_mode(opt)
    model = SIPN(opt.net, opt.pre_model)

    # Load the dataset
    dataset = PersonSearchDataset(opt.data_dir)

    save_dir = opt.out_dir
    print('Trained models will be save to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Choose parameters to be updated during training
    lr = opt.lr
    params = []
    print('These parameters will be updated during training:')
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            print(key)
            # TODO: set different decay for weight and bias
            params += [{'params': [value], 'lr': lr, 'weight_decay': 1e-4}]

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    elif opt.optimizer == 'Adam':
        lr *= 0.1
        optimizer = torch.optim.Adam(params)
    else:
        raise KeyError(opt.optimizer)

    # TODO: add resume

    # Train the model
    train_model(dataset, model, lr, optimizer, opt.epochs, use_cuda, save_dir)


if __name__ == '__main__':

    main()
