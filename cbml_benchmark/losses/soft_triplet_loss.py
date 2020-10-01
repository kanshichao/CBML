import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

from cbml_benchmark.losses.registry import LOSS

@LOSS.register('softtriple_loss')
class SoftTriple(nn.Module):
    def __init__(self, cfg):
        super(SoftTriple, self).__init__()
        self.la = cfg.LOSSES.SOFTTRIPLE_LOSS.LA
        self.gamma = 1./cfg.LOSSES.SOFTTRIPLE_LOSS.GAMMA
        self.tau = cfg.LOSSES.SOFTTRIPLE_LOSS.TAU
        self.margin = cfg.LOSSES.SOFTTRIPLE_LOSS.MARGIN
        self.cN = cfg.LOSSES.SOFTTRIPLE_LOSS.CLUSTERS
        self.K = cfg.LOSSES.SOFTTRIPLE_LOSS.K
        self.fc = Parameter(torch.Tensor(cfg.MODEL.HEAD.DIM, self.cN*self.K)).cuda()
        self.weight = torch.zeros(self.cN*self.K, self.cN*self.K, dtype=torch.bool).cuda()
        for i in range(0, self.cN):
            for j in range(0, self.K):
                self.weight[i*self.K+j, i*self.K+j+1:(i+1)*self.K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
