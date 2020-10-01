import torch
from torch import nn

from cbml_benchmark.losses.registry import LOSS


@LOSS.register('crossentropy_loss')
class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, labels):
        loss = self.crossentropy_loss(predicts,labels)
        return loss
