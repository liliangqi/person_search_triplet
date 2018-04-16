# -----------------------------------------------------
# Custom Losses used in Person Search Architecture
#
# Author: Liangqi Li and Tong Xiao
# Creating Date: Jan 3, 2018
# Latest rectifying: Apr 15, 2018
# -----------------------------------------------------
import yaml

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import numpy as np


def smooth_l1_loss(bbox_pred, bbox_info, sigma=1., dim=[1]):
    sigma_2 = sigma ** 2
    bbox_targets, bbox_inside_weights, bbox_outside_weights = bbox_info
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two Variable matrices.
    ---
    param:
        x: PyTorch Variable with shape (m, d)
        y: PyTorch Variable with shape (n, d)
    return:
        distance: PyTorch Variable with shape (m, n)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    distance = xx + yy
    distance.addmm_(1, -2, x, y.t())
    distance = distance.clamp(min=1e-12).sqrt()

    return distance


def triplet_loss(q_feature, q_label, g_features, g_labels, mode='hard'):
    """
    Compute triplet loss between query and galleries using query as anchor.
    ---
    param:
        q_feature: Pytorch Variable refers to query feature with shape (1, D)
        g_features: Pytorch Variable refers to gallery features with shape
                    (N, D), where N is the number of persons in the gallery
        q_label: (int) the label of the query
        g_label: Pytorch Variable refers to labels gallery rois
        mode: (str) hard mining or average
    """
    assert q_label in g_labels.data, \
        '{:d} does not exist in gallery rois'.format(q_label)
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    loss_model = nn.MarginRankingLoss(margin=config['triplet_margin'])

    # TODO: add cosine similarity
    distance = euclidean_distance(q_feature, g_features)
    is_pos = g_labels == q_label
    is_neg = g_labels != q_label

    if mode == 'hard':
        dist_ap = torch.max(distance[is_pos])
        dist_an = torch.min(distance[is_neg])
        loss = loss_model(dist_an, dist_ap, Variable(torch.ones(1, 1)))
        return loss

    elif mode == 'average':
        dist_ap = torch.mean(distance[is_pos])
        dist_an = torch.mean(distance[is_neg])
        loss = loss_model(dist_an, dist_ap, Variable(torch.ones(1, 1)))
        return loss

    else:
        raise KeyError(mode)


class OIM(Function):
    def __init__(self, lut, queue, momentum):
        super(OIM, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum  # TODO: use exponentially weighted average

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
                self.queue[:, :] = tmp[:, :]
            elif y < len(self.lut):
                self.lut[y] = self.momentum * self.lut[y] + \
                              (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            else:
                continue

        return grad_inputs, None


def oim_loss(reid_feat, aux_label, num_pid, q_size, lut, queue, momentum=0.5):
    aux_label_np = aux_label.data.cpu().numpy()
    invalid_inds = np.where((aux_label_np < 0) | (aux_label_np >= num_pid))
    aux_label_np[invalid_inds] = -1
    pid_label = Variable(torch.from_numpy(aux_label_np).long().cuda()).view(-1)
    aux_label = aux_label.view(-1)

    reid_result = OIM(lut, queue, momentum)(reid_feat, aux_label)
    reid_loss_weight = torch.cat([torch.ones(num_pid).cuda(),
                                  torch.zeros(q_size).cuda()])
    scalar = 10
    reid_loss = F.cross_entropy(reid_result * scalar, pid_label,
                                weight=reid_loss_weight, ignore_index=-1)
    return reid_loss
