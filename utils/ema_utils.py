import copy

import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay=0.999) -> None:
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
