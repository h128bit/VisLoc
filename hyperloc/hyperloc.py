from collections.abc import Callable

from torch import nn 

class HyperLoc(nn.Module):
    def __init__(self, 
                 embbender: nn.Module, 
                 target_net_generator: Callable[[], list[nn.Module]],
                 ghn: nn.Module):
        super().__init__()

        self.embbender = embbender 
        self.target_net_generator = target_net_generator 
        self.ghn = ghn


    def forward(self, x):
        embbeds = self.embbender(x)

        tnets = self.target_net_generator()

        archs = self.ghn(tnets, embbeds)

        return archs

