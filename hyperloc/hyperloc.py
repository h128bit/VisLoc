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
        print(f"archs num in forward {len(tnets)}")
        print(f"x in forward {x.shape}")
        embbeds = self.embbender(x)
        print(f"embbeds in forward {embbeds}")
        archs = self.ghn(tnets, embbeds)

        return archs

