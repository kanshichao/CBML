import torch
import torch.nn
import warnings
warnings.filterwarnings("ignore")

from dbml_benchmark.losses.registry import LOSS


class GradRev(torch.autograd.Function):

    def forward(self, x):
        return x.view_as(x)

    def backward(self,grad_output):
        return (grad_output*-1.)


def grad_reverse(x):
    return GradRev()(x)


@LOSS.register('adv_loss')
class AdvLoss(torch.nn.Module):

    def __init__(self, cfg):
        super(AdvLoss, self).__init__()
        self.class_dim, self.aux_dim = cfg.LOSSES.ADV_LOSS.CLASS_DIM, cfg.LOSSES.ADV_LOSS.AUX_DIM
        self.proj_dim = cfg.LOSSES.ADV_LOSS.PROJ_DIM
        self.regressor = torch.nn.Sequential(torch.nn.Linear(self.aux_dim,self.proj_dim),torch.nn.Linear(self.proj_dim,self.class_dim)).type(torch.cuda.FloatTensor)

    def forward(self, class_feature, aux_feature):
        features = [torch.nn.functional.normalize(grad_reverse(class_feature),dim=-1),torch.nn.functional.normalize(grad_reverse(aux_feature),dim=-1)]
        sim_loss = -1.*torch.mean(torch.mean((features[0]*torch.nn.functional.normalize(self.regressor(features[1]),dim=-1))**2,dim=-1))
        return sim_loss

