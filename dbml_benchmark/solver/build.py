import torch

from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, loss_param=None):
    params = []
    lr_mul = 1.0
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "backbone" in key:
            lr_mul = 0.1
        params += [{"params": [value], "lr_mul": lr_mul}]
    if loss_param is not None: # for soft triplet loss
        params += [{"params": loss_param.parameters(), "lr_mul": lr_mul}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params,
                                                                lr=cfg.SOLVER.BASE_LR,
                                                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
