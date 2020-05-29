
from .dbml import DBMLLoss
from .registry import LOSS


def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not registered in registry :{LOSS.keys()}'
    return LOSS[loss_name](cfg)
