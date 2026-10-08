import os
import re
import json
from pathlib import Path
from typing import Any

import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset



def coords_to_pixel(point: list[int|float]|tuple[int|float, int|float],  
                    map_resolution: list[float]|tuple[float, float], 
                    map_origin: list[float]|tuple[float, float]) -> tuple[int, int]:
    x_meters, y_meters = point
    
    x = int((x_meters - map_origin[0]) / map_resolution[0])
    y = int((y_meters - map_origin[1]) / map_resolution[1])

    return x, y


def correct_coord_after_map_resize(point: list[int|float]|tuple[int|float, int|float], 
                                   resize: int, 
                                   image_size: list[int]|tuple[int, int], 
                                   map_resolution: list[float]|tuple[float, float], 
                                   map_origin: list[float]|tuple[float, float]) -> tuple[int, int]:
    x_meters, y_meters = coords_to_pixel(point, map_resolution, map_origin)

    x = int((x_meters - map_origin[0]) / map_resolution[0])
    y = int((y_meters - map_origin[1]) / map_resolution[1]) 

    w, h = image_size

    scale_x = resize / w
    scale_y = resize / h

    x = int(x * scale_x)
    y = int(y * scale_y)

    return x, y



class _AnyVisLocPart(Dataset):
    def __init__(self, 
                 data: list[dict], 
                 satellite_resize: int, 
                 image_resize: int):
        super().__init__()

        self.data = data 
        self.satellite_resize = satellite_resize
        self.image_resize = image_resize

        self.image_compose = A.Compose([
            A.ColorJitter(
                brightness_range=(0.7, 1.3),  
                contrast_range=(0.7, 1.3),
                saturation_range=(0.8, 1.3),
                hue_range=(-0.05, 0.05),
                p=0.15
            ),

            A.OneOf([
                A.Blur(blur_range=(3, 7), p=1.0),
                A.GaussianBlur(blur_range=(3, 7), p=1.0),
                A.MotionBlur(blur_range=(3, 7), p=1.0),
                A.MedianBlur(blur_range=(3, 7), p=1.0),
                A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=1.0),
            ], p=0.15),

            A.RandomSizedCrop(
                min_max_height=(100, 400),  
                size=(self.image_resize, self.image_resize),
                w2h_ratio=1.0, 
                p=0.2
            ),

            A.Resize(height=self.image_resize, 
                     width=self.image_resize, 
                     interpolation=cv2.INTER_LINEAR, 
                     p=1.0),

            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),   
                max_pixel_value=255.0, 
                p=1.0
            ),

            ToTensorV2()
        ])

        self.satellite_image_compose = A.Compose([
            A.ColorJitter(
                brightness_range=(0.85, 1.1),  
                contrast_range=(0.85, 1.1),
                saturation_range=(0.85, 1.1),
                hue_range=(-0.02, 0.02),
                p=0.1
            ),

            A.RandomRotate90(p=0.15),

            A.Resize(height=self.satellite_resize, 
                    width=self.satellite_resize, 
                    interpolation=cv2.INTER_LINEAR, 
                    p=1.0),

            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),   
                max_pixel_value=255.0, 
                p=1.0
            ),

            ToTensorV2()
        ], keypoint_params=A.KeypointParams(coord_format='xy'))


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index) -> Any:
        npz_file, meta = self.data[index]

        print(npz_file)

        root = Path(meta["root_path"])

        file = np.load(root / npz_file)
        image = np.transpose(file["image"], (1, 0, 2)) #  from whc to hwc format
        point = file["xyz"][0:2]

        satellite_image = np.array(Image.open(root / meta["satellite_path"]))
        satellite_image = np.transpose(satellite_image, (1, 0, 2)) #  from whc to hwc format
        map_resolution = meta["map_resolution"]
        map_origin = meta["map_origin"]

        point = coords_to_pixel(point, map_resolution, map_origin)
        point = np.array([point])

        image = self.image_compose(image=image)["image"]
        res = self.satellite_image_compose(image=satellite_image, keypoints=point)
        sat_image = res["image"]
        point = torch.tensor(res["keypoints"][0]) / self.satellite_resize

        return point, sat_image, image


class AnyVisLocDataset:
    def __init__(self, 
                 path_to_scenes: list[str|os.PathLike|Path], 
                 satellite_resize: int,
                 image_resize: int, 
                 split_on_train_test: bool=True,
                 split_ratio: float=0.7):
        
        self.scenes_data = list()
        for scene_path in path_to_scenes:
            scene_path = Path(scene_path)
            scene_files = os.listdir(scene_path)
            pattern = re.compile(r".+reference\.json")
            reference = [s for s in scene_files if pattern.search(s)][0]
            pattern = re.compile(r".+\.npz")
            scene_name = scene_path.name
            files = [s for s in scene_files if pattern.search(s)]

            with open(scene_path / reference, "r", encoding="utf-8") as f:
                ref = f.read()
            ref_data = json.loads(ref)
            satellite_info = ref_data['modes']['satellite']

            d = {
                "scene_name": scene_name,
                "map_resolution": satellite_info['map_resolution'],
                "map_origin": satellite_info['map_origin_local'],
                "satellite_path": satellite_info["map_path"],
                "size": satellite_info["map_size"],
                "root_path": scene_path
            }
            files = [(item, d) for item in files]
            self.scenes_data.extend(files)

        if not split_on_train_test:
            split_ratio = 1.0

        ratio = int(len(self.scenes_data) * split_ratio) 
        self.train_part = self.scenes_data[0:ratio:]
        self.test_part = self.scenes_data[ratio::]

        self.train_part = _AnyVisLocPart(self.train_part, satellite_resize, image_resize)
        self.test_part = _AnyVisLocPart(self.test_part, satellite_resize, image_resize)


    def get_train_part(self) -> _AnyVisLocPart:
        return self.train_part


    def get_test_part(self) -> _AnyVisLocPart:
        return self.test_part