# encoding: utf-8
"""
@author:
@contact:
"""
import torch
from torch import nn

from dbml_benchmark.losses.registry import LOSS


class Normalizing(object):
    def __init__(self):
        super(Normalizing, self).__init__()
    def normalize_rank(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x


class EucDist(object):
    def __init__(self):
        super(EucDist, self).__init__()
    def euclidean_dist_rank(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


class RankLoss(object):
    def __init__(self):
        super(RankLoss, self).__init__()

    def rank_loss(self, dist_mat, labels, margin, alpha, tval):
        """
        Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]

        """
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        total_loss = 0.0
        for ind in range(N):
            is_pos = labels.eq(labels[ind])
            is_pos[ind] = 0
            is_neg = labels.ne(labels[ind])

            dist_ap = dist_mat[ind][is_pos]
            dist_an = dist_mat[ind][is_neg]

            ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
            ap_pos_num = ap_is_pos.size(0) + 1e-5
            ap_pos_val_sum = torch.sum(ap_is_pos)
            loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

            an_is_pos = torch.lt(dist_an, alpha)
            an_less_alpha = dist_an[an_is_pos]
            an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
            an_weight_sum = torch.sum(an_weight) + 1e-5
            an_dist_lm = alpha - an_less_alpha
            an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
            loss_an = torch.div(an_ln_sum, an_weight_sum)

            total_loss = total_loss + loss_ap + loss_an
        total_loss = total_loss * 1.0 / N
        return total_loss


@LOSS.register('rank_loss')
class RankedListLoss(nn.Module):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self, cfg):
        super(RankedListLoss, self).__init__()
        self.margin = cfg.LOSSES.RANKED_LIST_LOSS.MARGIN
        self.alpha = cfg.LOSSES.RANKED_LIST_LOSS.ALPHA
        self.margin_gnn = cfg.LOSSES.RANKED_LIST_LOSS.MARGIN_GNN
        self.alpha_gnn = cfg.LOSSES.RANKED_LIST_LOSS.ALPHA_GNN
        self.tval = cfg.LOSSES.RANKED_LIST_LOSS.TVAL
        self.knn = cfg.DATA.KNN
        self.batch_size = cfg.DATA.TRAIN_BATCHSIZE
        self.normalize = Normalizing()
        self.eucdist = EucDist()
        self.rankloss = RankLoss()


    def forward(self, feats, labels,knn_opt=False, gnn=False):
        #feats = self.normalize.normalize_rank(feats, axis=-1)
        if knn_opt:
            loss = 0
            for i in range(self.batch_size):
                dist_mat = self.eucdist.euclidean_dist_rank(feats[i*self.knn:(i+1)*self.knn], feats[i*self.knn:(i+1)*self.knn])
                if gnn:
                    loss += self.rankloss.rank_loss(dist_mat, labels[i * self.knn:(i + 1) * self.knn], self.margin_gnn,
                                                    self.alpha_gnn, self.tval)
                else:
                    loss += self.rankloss.rank_loss(dist_mat, labels[i*self.knn:(i+1)*self.knn], self.margin, self.alpha, self.tval)
            loss = loss / self.batch_size
        else:
            dist_mat = self.eucdist.euclidean_dist_rank(feats,feats)
            if gnn:
                loss = self.rankloss.rank_loss(dist_mat, labels, self.margin_gnn, self.alpha_gnn, self.tval)
            else:
                loss = self.rankloss.rank_loss(dist_mat, labels, self.margin, self.alpha, self.tval)
        return loss

# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.
#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.
#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
#         if self.use_gpu: targets = targets.cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (- targets * log_probs).mean(0).sum()
#         return loss
#
# @LOSS.register('ranked_softmax_loss')
# class Ranked_SoftMax_Loss(object):
#     def __init__(self, margin=1.3, alpha=2.0, tval=1.0):
#         self.margin = margin
#         self.alpha = alpha
#         self.tval = tval
#         self.ranked_loss = RankLoss()
#         self.xent = CrossEntropyLabelSmooth(num_classes=100)
#
#     def __call__(self, feat, labels):
#         loss = self.xent(labels, labels) + 0.4 * self.ranked_loss(feat, labels)[0]
#         return loss