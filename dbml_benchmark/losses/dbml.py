import torch
from torch import nn

from dbml_benchmark.losses.registry import LOSS


@LOSS.register('dbml_loss')
class DBMLLoss(nn.Module):
    def __init__(self, cfg):
        super(DBMLLoss, self).__init__()
        self.pos_a = cfg.LOSSES.DBML_LOSS.POS_A
        self.pos_b = cfg.LOSSES.DBML_LOSS.POS_B
        self.neg_a = cfg.LOSSES.DBML_LOSS.NEG_A
        self.neg_b = cfg.LOSSES.DBML_LOSS.NEG_B
        self.margin = cfg.LOSSES.DBML_LOSS.MARGIN
        self.weight = cfg.LOSSES.DBML_LOSS.WEIGHT
        self.adaptive_neg = cfg.LOSSES.DBML_LOSS.ADAPTIVE_NEG

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):

            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            mean_ = torch.mean(sim_mat[i])
            sigma_ = torch.mean(torch.sum(torch.pow(sim_mat[i]-mean_,2)))

            pp = pos_pair_ - self.margin < max(neg_pair_)
            pos_pair = pos_pair_[pp]
            if self.adaptive_neg:
                np = neg_pair_ + self.margin > min(pos_pair_)
                neg_pair = neg_pair_[np]
            else:
                np = torch.argsort(neg_pair_)
                neg_pair = neg_pair_[np[-100:]]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            mean = (torch.sum(pos_pair) + torch.sum(neg_pair)) / (len(pos_pair) + len(neg_pair))
            sigma = (torch.sum(torch.pow(pos_pair-mean,2))+torch.sum(torch.pow(neg_pair-mean,2)))/(len(pos_pair) + len(neg_pair))

            fp = 1. + torch.sum(torch.exp(-1./self.pos_b * (pos_pair - self.pos_a)))
            fn = 1. + torch.sum(torch.exp( 1./self.neg_b * (neg_pair - self.neg_a)))
            pos_loss = torch.log(fp)
            neg_loss = torch.log(fn)
            pos_neg_loss = torch.abs(mean_-mean) + torch.abs(sigma_-sigma)
            loss.append((pos_loss + neg_loss + self.weight*pos_neg_loss))

        if len(loss) == 0:
            return torch.zeros(1, requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss