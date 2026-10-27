import os 
from pathlib import Path
from typing import Any, Callable

from torch import nn 
from torch.utils.data import Dataset



class ArchImageDataset(Dataset):
    def __init__(self, 
                 arch_generator: Callable[[], list[nn.Module]], 
                 data_part):
        
        super().__init__()

        self.image_dataset = data_part
        self.arch_generator = arch_generator

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index) -> Any:
        vals = self.image_dataset[index]
        arch = self.arch_generator()[0]
        return arch, *vals

