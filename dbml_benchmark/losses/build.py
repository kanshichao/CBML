
from .dbml import DBMLLoss
from .crossentropy_loss import CrossEntropyLoss
from .margin_loss import MarginLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .ranked_list_loss import RankedListLoss
from .soft_triplet_loss import SoftTriple
from .proxynca import ProxyNCA
from .npair_loss import NPairLoss
from .angular_loss import AngularLoss
from .contrastive_loss import ContrastiveLoss, OnlineContrastiveLoss
from .triplet_loss import TripletLoss, OnlineTripletLoss
from .cluster_loss import ClusterLoss, ClusterLoss_local
from .histogram_loss import HistogramLoss
from .center_loss import CenterLoss
from .registry import LOSS


def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not registered in registry :{LOSS.keys()}'
    return LOSS[loss_name](cfg)
