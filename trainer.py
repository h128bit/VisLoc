import pandas as pd

from ghn3 import GHN3
from datasets import AnyVisLocDataset

class TrainStep:
    def __init__(self, model, optimizer, criterion, device):
        