import torch
from torch.nn import Parameter
import torch.nn.functional as F
from cbml_benchmark.losses.registry import LOSS


class BinarizeLabels(object):
    def __init__(self):
        super(BinarizeLabels, self).__init__()

    def binarize_and_smooth_labels(self, T, nb_classes, smoothing_const=0.1):
        # Optional: BNInception uses label smoothing, apply it for retraining also
        # "Rethinking the Inception Architecture for Computer Vision", p. 6
        import sklearn.preprocessing
        T = T.cpu().numpy()
        T = sklearn.preprocessing.label_binarize(
            T, classes=range(0, nb_classes)
        )
        T = T * (1 - smoothing_const)
        T[T == 0] = smoothing_const / (nb_classes - 1)
        T = torch.FloatTensor(T).cuda()
        return T

@LOSS.register('proxynca_loss')
class ProxyNCA(torch.nn.Module):
    def __init__(self,
        cfg
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.nb_classes = cfg.LOSSES.PROXY_LOSS.NB_CLASSES
        self.sz_embedding = cfg.MODEL.HEAD.DIM
        self.proxies = Parameter(torch.randn(self.nb_classes, self.sz_embedding) / 8)
        self.smoothing_const = cfg.LOSSES.PROXY_LOSS.SMOOTHING_CONST
        self.scaling_x = cfg.LOSSES.PROXY_LOSS.SCALING_X
        self.scaling_p = cfg.LOSSES.PROXY_LOSS.SCALING_P
        self.binarizelabel = BinarizeLabels()

    def forward(self, X, T):
        P = F.normalize(self.proxies, p = 2, dim = -1).cuda() * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1).cuda() * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = self.binarizelabel.binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()
