import torch
import torch.nn as nn


class EMA():
    def __init__(self,model: nn.Module , decay : float) :
        self.decay = decay
        self.step = 0
        self.shadow = {name : p.detach().clone() for name, p in model.named_parameters()}


    def warmup(self,step) :
        d =min(self.decay, (1+step)/ (10+ step))
        return d

    @torch.no_grad()
    def update(self,model):
        self.step+=1
        d = self.warmup(self.step)
        for name,p in model.named_parameters():
            self.shadow[name] = self.shadow[name] * d + (1-d)*p.detach()

    @torch.no_grad()        
    def copy_to(self, model):
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name])
            