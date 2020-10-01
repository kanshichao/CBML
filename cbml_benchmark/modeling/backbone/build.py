from cbml_benchmark.modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .googlenet import GoogLeNet


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME in BACKBONES, \
        f"backbone {cfg.MODEL.BACKBONE} is not registered in registry : {BACKBONES.keys()}"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME]()
