import torch
import ignite.distributed as idist

from ghn3 import GHN3
from hyperloc import HyperVisLocTrainer, HyperLoc, generate_resnet18, get_model
from datasets import AnyVisLocDataset


def main(local_rank, config):
    print("========| BUILD DATASET")
    dataset = AnyVisLocDataset(path_to_scenes=config["path_to_scenes"], 
                               satellite_resize=config["sat_image_size"], 
                               image_resize=config["image_size"])
    
    train_loader = torch.utils.data.DataLoader(dataset.get_train_part(), batch_size=config["batch_size"], num_workers=4)
    eval_loader = torch.utils.data.DataLoader(dataset.get_test_part(), batch_size=config["batch_size"], num_workers=4)

    print("========| BUILD MODELS")
    ghn = GHN3(**config["ghn_config"])
    emb_model = get_model("resnet200", config["ghn_config"]["hid"])
    model = HyperLoc(embbender=emb_model, 
                     target_net_generator=generate_resnet18(config["batch_size"]), 
                     ghn=ghn)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.MSELoss()

    print("========| BUILD TRAINER")
    trainer = HyperVisLocTrainer(
        model=model,
        model_generator=generate_resnet18(1),
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        scheduler=scheduler,
        sat_image_size=config["sat_image_size"],
        log_path=config["log_path"]
    )

    print("========| START TRAINING")
    trainer.trainer_engine.run(trainer.train_dataloader, max_epochs=config["max_epochs"])


if __name__ == "__main__":
    path_to_train_scenes = [r"C:\Users\t.asmus\Desktop\data\AnyVisLoc\Scene_10\Scene_10"]
    ghn_config = {
        "max_shape": (1000, 1000, 10, 10),
        "num_classes": 2,
        "hid": 80
    }
    config = {
        "ghn_config": ghn_config,
        "batch_size": 1,   
        "lr": 1e-3,
        "max_epochs": 1,
        "sat_image_size": 256,
        "image_size": 64,
        "log_path": "./tests/runs",
        "path_to_scenes": path_to_train_scenes
    }

    # Запуск через idist.Parallel. 
    # Если запустить как 'python main.py', это отработает как обычное обучение на 1 GPU.
    # Если запустить через 'torchrun', это автоматически масштабируется на все доступные GPU.
    with idist.launcher.Parallel(backend=None) as parallel:
        parallel.run(main, config)