# Copyright (c) Company Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np

from dbml_benchmark.losses.registry import LOSS


@LOSS.register('ms_loss')
class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh_pos = 0.5
        self.thresh_neg = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.hard_mining = cfg.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING

    def forward(self, feats, labels, memory_feats = None, memory_label = None):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        if memory_feats is None or len(memory_feats)==0:
            sim_mat = torch.matmul(feats, torch.t(feats))
        else:
            sim_mat = torch.matmul(memory_feats, torch.t(feats)).transpose(0,1)

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if memory_feats is None or len(memory_feats)==0:
                pos_pair_ = sim_mat[i][labels == labels[i]]
                pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
                neg_pair_ = sim_mat[i][labels != labels[i]]
            else:
                pos_pair_ = sim_mat[i][memory_label == labels[i]]
                pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
                neg_pair_ = sim_mat[i][memory_label != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            if self.hard_mining:
                np = neg_pair_ + self.margin > torch.min(pos_pair_)
                pp = pos_pair_ - self.margin < torch.max(neg_pair_)
                neg_pair = neg_pair_[np]
                pos_pair = pos_pair_[pp]
            else:
                pos_pair = pos_pair_
                neg_pair = neg_pair_

            # neg_pair = neg_pair_[neg_pair_>0.3]
            # pos_pair = pos_pair_

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step

            pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh_pos))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh_neg))))
            loss.append(neg_loss+pos_loss)

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss
