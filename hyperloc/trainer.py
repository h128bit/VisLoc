import gc
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, stack_module_state
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, global_step_from_engine


class HyperVisLocTrainer:
    def __init__(self, 
                 model, 
                 model_generator, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 eval_dataloader, 
                 batch_size=4, 
                 num_workers=4, 
                 scheduler=None, 
                 sat_image_size=1, 
                 log_path="run"):
        # consts
        self.device = idist.device() # device
        self.sat_image_size = sat_image_size

        # print(self.device)

        # general setup
        self.model = model
        self.model.to(self.device)
        self.model = idist.auto_model(self.model)

        self.criterion = criterion
        self.optimizer = idist.auto_optim(optimizer)
        self.scheduler = scheduler

        self.train_dataloader = idist.auto_dataloader(train_dataloader, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        self.eval_dataloader = idist.auto_dataloader(eval_dataloader, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        self.model_generator = model_generator
        self.target_model_template = self.model_generator(self.device)[0].eval()
        for param in self.target_model_template.parameters():
            param.requires_grad = False

        # init engines
        self.trainer_engine = Engine(self.train_step)
        self.eval_engine = Engine(self.eval_step)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.step_scheduler)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.run_eval)

        # setup metrics
        _output_transform = lambda y: (y["y_pred"], y["y_true"])
        MeanSquaredError(output_transform=_output_transform).attach(self.eval_engine, "mse")
        MeanAbsoluteError(output_transform=_output_transform).attach(self.eval_engine, "mae")

        # setup logging
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, self.log_training_loss)
        self.eval_engine.add_event_handler(Events.COMPLETED, self.log_metrics)

        if idist.get_rank() == 0:
            self.log_path = Path(log_path)
            i = 1
            while True:
                if self.log_path.exists():
                    self.log_path = Path(str(self.log_path) + f"_{i}")
                    i += 1
                else:
                    break
            self.log_path.mkdir(parents=True, exist_ok=True)

            self.train_columns = ["epoch", "iteration", "value"]
            self.eval_columns = ["epoch", "iteration", "mse", "mae"]
            self._path_to_train_table = self.log_path / "train_loss.csv"
            self._path_to_eval_table = self.log_path / "eval_metrics.csv"
            pd.DataFrame(columns=self.train_columns).to_csv(self._path_to_train_table, columns=self.train_columns, index=False)
            pd.DataFrame(columns=self.eval_columns).to_csv(self._path_to_eval_table, columns=self.eval_columns, index=False)

        # setup checkpoints
        self._score_function = lambda engine: engine.state.metrics["mse"]
        modal_checkpoint = ModelCheckpoint(
            self.log_path / "checkpoint",
            n_saved=1,
            filename_prefix="best",
            score_function=self._score_function,
            score_name="mse",
            global_step_transform=global_step_from_engine(self.trainer_engine), # helps fetch the trainer's state
            )
        self.eval_engine.add_event_handler(Events.COMPLETED, modal_checkpoint, {"model": self.model})


    def _forward_single(self, params, buffers, img):
        return functional_call(self.target_model_template, (params, buffers), img.unsqueeze(0))


    def _batch_to_device(self, batch):
        points, sat_images, images = batch

        points = points.to(self.device)
        sat_images = sat_images.to(self.device)
        images = images.to(self.device)

        return points, sat_images, images


    def _step(self, batch):
        points, sat_images, images = self._batch_to_device(batch)

        archs = self.model_generator(self.device)

        predicted_archs = self.model(archs, sat_images)

        stacked_params, stacked_buffers = stack_module_state(predicted_archs)


        params_in_dims = {k: 0 for k in stacked_params.keys()}
        buffers_in_dims = {k: 0 for k in stacked_buffers.keys()}

        vvmap = vmap(
            self._forward_single,
            in_dims=(params_in_dims, buffers_in_dims, 0)
            )
        pred_points = vvmap(stacked_params, stacked_buffers, images)

        return pred_points.squeeze(-1), points


    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        pred_points, target_points = self._step(batch)

        loss = self.criterion(pred_points, target_points)
        loss.backward()

        self.optimizer.step()

        if engine.state.iteration % 10 == 0: 
            print("========")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print("========")

        torch.cuda.empty_cache()
        gc.collect()

        return loss.item()


    def eval_step(self, engine, batch):
        self.model.eval() 

        with torch.no_grad():
            pred_points, points = self._step(batch)

        # normalizet coordinates to original
        pred_points *= self.sat_image_size
        points *= self.sat_image_size 

        return {"y_pred": pred_points, "y_true": points}


    def run_eval(self, engine):
        self.eval_engine.run(self.eval_dataloader)


    def step_scheduler(self, engine):
        if self.scheduler:
            self.scheduler.step()

    def _write_in_csv_table(self, path, data):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)


    def log_training_loss(self, engine):
        if idist.get_rank() == 0:
            epoch = engine.state.epoch
            iteration = engine.state.iteration
            value = engine.state.output
            data = [epoch, iteration, value]

            print(f"Epoch[{epoch}], Iter[{iteration}] Loss: {value:.5f}")

            self._write_in_csv_table(self._path_to_train_table, data)


    def log_metrics(self, engine):
        if idist.get_rank() == 0:
            epoch = engine.state.epoch
            iteration = engine.state.iteration
            mse = engine.state.metrics["mse"]
            mae = engine.state.metrics["mae"]
            data = [epoch, iteration, mse, mae]

            print(f"Epoch[{epoch}], Iter[{iteration}] MSE: {mse:.5f} MAE: {mae:.5f}")

            self._write_in_csv_table(self._path_to_eval_table, data)





    
