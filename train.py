import torch
import ignite.distributed as idist

from ghn3 import GHN3
from hyperloc import HyperVisLocTrainer, HyperLoc, generate_resnet18,  get_model
from datasets import AnyVisLocDataset

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def main(local_rank, config):
    is_main_rank = idist.get_rank() == 0
    print("========| BUILD DATASET")
    dataset = AnyVisLocDataset(path_to_scenes=config["path_to_scenes"], 
                               satellite_resize=config["sat_image_size"], 
                               image_resize=config["image_size"],
                               tile_size=config["tile_size"],
                               do_sat_tiff_convert=is_main_rank)
    idist.barrier()
    train_dataset = dataset.get_train_part()
    eval_dataset = dataset.get_test_part()

    print("========| BUILD MODELS")
    ghn = GHN3(**config["ghn_config"])
    emb_model = get_model("resnet200", config["ghn_config"]["hid"])
    model = HyperLoc(embbender=emb_model, 
                     ghn=ghn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"], eta_min=1e-6)
    criterion = torch.nn.MSELoss()

    print("========| BUILD TRAINER")
    trainer = HyperVisLocTrainer(
        model=model,
        model_generator=generate_resnet18(config["batch_size"]),
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataset,
        eval_dataloader=eval_dataset,
        batch_size=config["batch_size"], 
        num_workers=4,
        scheduler=scheduler,
        sat_image_size=config["sat_image_size"],
        log_path=config["log_path"]
    )

    print("========| START TRAINING")
    trainer.trainer_engine.run(trainer.train_dataloader, max_epochs=config["max_epochs"])


if __name__ == "__main__":
    import json 

    path_to_config = "/workspace/tmp/visloc_train_config.json"
    with open(path_to_config, "r", encoding="utf-8") as f:
        s = f.read()
        config = json.loads(s)

    # path_to_train_scenes = [
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_01",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_02",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_03",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_04",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_05",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_06",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_07",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_08",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_09",
    #     "/workspace/tmp/dataset/anyvisloc/scenes/Scene_10",
    #     ]
    # ghn_config = {
    #     "max_shape": (1000, 1000, 10, 10),
    #     "num_classes": 2,
    #     "hid": 384
    # }
    # config = {
    #     "ghn_config": ghn_config,
    #     "batch_size": 8,   
    #     "lr": 1e-4,
    #     "max_epochs": 100,
    #     "sat_image_size": 1536,
    #     "image_size": 384,
    #     "log_path": "./skip-visloc_runs/run",
    #     "path_to_scenes": path_to_train_scenes,
    #     "tile_size": 2992
    # }

    with idist.Parallel(backend=config["backend"]) as parallel:
        parallel.run(main, config)