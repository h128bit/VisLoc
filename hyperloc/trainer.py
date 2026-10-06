import torch
from torch.func import functional_call, vmap, stack_module_state

from hyperloc.hyperloc import HyperLoc


class HyperVisLocTrainer:
    def __init__(self, emb_model, ghn, model_generator, criterion, optimizer, sheduler, device):
        self.device = device

        self.model = HyperLoc(emb_model, 
                              model_generator, 
                              ghn)
        self.model.to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer 
        self.sheduler = sheduler

        self.model_generator = model_generator

        self._compute_single_forward = self._init_single_forward()
        self._models_batch_forward = vmap(self._compute_single_forward, in_dims=(0, 0, 0))


    def _init_single_forward(self):
        model = self.model_generator()[0].to(self.device)

        def compute_single_forward(params, buffers, x):
            return functional_call(model, (params, buffers), x)
        
        return compute_single_forward


    def train_step(self, engine, batch):
        self.model.train()

        points, sat_images, images = batch
        points = points.to(self.device)
        sat_images = sat_images.to(self.device)
        images = images.to(self.device)
 
        archs = self.model(sat_images)

        batched_arch_params = stack_module_state(archs)

        pred_points = self._models_batch_forward(*batched_arch_params, images)

        loss = self.criterion(pred_points, points).mean()
        loss.backward()

        self.optimizer.step()

        return loss.item()
