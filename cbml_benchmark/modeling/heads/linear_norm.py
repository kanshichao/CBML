
import torch
from torch import nn

from cbml_benchmark.modeling.registry import HEADS
from cbml_benchmark.utils.init_methods import weights_init_classifier,weights_init_kaiming


@HEADS.register('linear_norm')
class LinearNorm(nn.Module):
    def __init__(self, cfg, in_channels):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, cfg.MODEL.HEAD.DIM)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
