from collections.abc import Callable

from torch import nn 

class HyperLoc(nn.Module):
    def __init__(self, 
                 embbender: nn.Module, 
                 ghn: nn.Module):
        super().__init__()

        self.embbender = embbender 
        self.ghn = ghn


    def forward(self, tnets, x):
        embbeds = self.embbender(x)

        archs = self.ghn(tnets, embbeds)

        return archs

