
from .dbml import DBMLLoss
from .crossentropy_loss import CrossEntropyLoss
from .margin_loss import MarginLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .ranked_list_loss import RankedListLoss
from .soft_triplet_loss import SoftTriple
from .registry import LOSS


def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not registered in registry :{LOSS.keys()}'
    return LOSS[loss_name](cfg)
