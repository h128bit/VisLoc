import os
import re
import json
from pathlib import Path
from typing import Any

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from datasets.utils.tiff_tools import *



def coords_to_pixel(point: list[int|float]|tuple[int|float, int|float],  
                    map_resolution: list[float]|tuple[float, float], 
                    map_origin: list[float]|tuple[float, float]) -> tuple[int, int]:
    x_meters, y_meters = point
    
    x = int((x_meters - map_origin[0]) / map_resolution[0])
    y = int((y_meters - map_origin[1]) / map_resolution[1])

    return x, y


def prepare_sattelite_image( 
                            scene_path: str|os.PathLike|Path, 
                            sat_path: str, 
                            tile_size: int=1024) -> str:
    
    sat_path_tiff = sat_path.split(".")[0] + ".tiff"

    conver_image_to_tiff(scene_path/sat_path, scene_path/sat_path_tiff, tile_size=tile_size)

    return sat_path_tiff



class _AnyVisLocPart(Dataset):
    def __init__(self, 
                 data: list[dict], 
                 satellite_resize: int, 
                 image_resize: int,
                 tile_size: int):
        super().__init__()

        self.data = data 
        self.satellite_resize = satellite_resize
        self.image_resize = image_resize
        self.tile_size = tile_size

        self.image_compose = A.Compose([
            # A.ColorJitter(
            #     brightness_range=(0.7, 1.3),  
            #     contrast_range=(0.7, 1.3),
            #     saturation_range=(0.8, 1.3),
            #     hue_range=(-0.05, 0.05),
            #     p=0.15
            # ),

            # A.OneOf([
            #     A.Blur(blur_range=(3, 7), p=1.0),
            #     A.GaussianBlur(blur_range=(3, 7), p=1.0),
            #     A.MotionBlur(blur_range=(3, 7), p=1.0),
            #     A.MedianBlur(blur_range=(3, 7), p=1.0),
            #     A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=1.0),
            # ], p=0.15),

            # A.RandomSizedCrop(
            #     min_max_height=(100, 400),  
            #     size=(self.image_resize, self.image_resize),
            #     w2h_ratio=1.0, 
            #     p=0.2
            # ),

            A.Resize(height=self.image_resize, 
                     width=self.image_resize, 
                     interpolation=cv2.INTER_LINEAR, 
                     p=1.0),

            # A.Normalize(
            #     mean=(0.485, 0.456, 0.406), 
            #     std=(0.229, 0.224, 0.225),   
            #     max_pixel_value=255.0, 
            #     p=1.0
            # ),

            ToTensorV2()
        ])

        self.satellite_image_compose = A.Compose([
            # A.ColorJitter(
            #     brightness_range=(0.85, 1.1),  
            #     contrast_range=(0.85, 1.1),
            #     saturation_range=(0.85, 1.1),
            #     hue_range=(-0.02, 0.02),
            #     p=0.1
            # ),

            # A.RandomRotate90(p=0.15),

            A.Resize(height=self.satellite_resize, 
                    width=self.satellite_resize, 
                    interpolation=cv2.INTER_LINEAR, 
                    p=1.0),

            # A.Normalize(
            #     mean=(0.485, 0.456, 0.406), 
            #     std=(0.229, 0.224, 0.225),   
            #     max_pixel_value=255.0, 
            #     p=1.0
            # ),

            ToTensorV2()
        ], keypoint_params=A.KeypointParams(coord_format='xy'))


    def __len__(self) -> int:
        return len(self.data)
        

    def __getitem__(self, index) -> Any:
        npz_file, meta = self.data[index]

        root = Path(meta["root_path"])

        file = np.load(root / npz_file)
        image = file["image"] 
        point = list(file["xyz"])[0:2]

        satellite_tiff_path = meta["satellite_tiff_path"]

        map_resolution = meta["map_resolution"]
        map_origin = meta["map_origin"]

        point = coords_to_pixel(point, map_resolution, map_origin)
        point = np.array([point])

        tile_index = point // self.tile_size ## point in format [ point ]
        satellite_tile = load_tile(root/satellite_tiff_path, tile_index[0])
        point = point - tile_index * self.tile_size # convert point to tile local coordinates

        image = self.image_compose(image=image)["image"]
        res = self.satellite_image_compose(image=satellite_tile, keypoints=point)
        sat_image = res["image"]
        point = torch.tensor(res["keypoints"][0]) #/ self.satellite_resize

        return point, sat_image, image


class AnyVisLocDataset:
    def __init__(self, 
                 path_to_scenes: list[str|os.PathLike|Path], 
                 satellite_resize: int,
                 image_resize: int, 
                 tile_size: int=1024,
                 split_on_train_test: bool=True,
                 split_ratio: float=0.7,
                 do_sat_tiff_convert=True):
        
        self.scenes_data = list()
        for scene_path in tqdm(path_to_scenes, desc="Prepare data: "):
            scene_path = Path(scene_path)
            scene_files = os.listdir(scene_path)
            pattern = re.compile(r".+reference\.json")
            reference = [s for s in scene_files if pattern.search(s)][0]
            pattern = re.compile(r".+\.npz")
            scene_name = scene_path.name
            npz_files = [s for s in scene_files if pattern.search(s)]

            with open(scene_path / reference, "r", encoding="utf-8") as f:
                ref = f.read()
            ref_data = json.loads(ref)
            satellite_info = ref_data['modes']['satellite']

            sattelite_scene_file_path = satellite_info["map_path"]
            if do_sat_tiff_convert:
                sat_path_tiff = prepare_sattelite_image(scene_path, sattelite_scene_file_path, tile_size)
            else: 
                sat_path_tiff = sattelite_scene_file_path.split(".")[0] + ".tiff"
        
            d = {
                "scene_name": scene_name,
                "map_resolution": satellite_info['map_resolution'],
                "map_origin": satellite_info['map_origin_local'],
                "satellite_path": sattelite_scene_file_path,
                "satellite_tiff_path": sat_path_tiff,
                "size": satellite_info["map_size"],
                "root_path": scene_path
            }

            files = []
            h, w = d["size"]
            for npz in npz_files:
                npz_ = np.load(scene_path / npz)
                point = list(npz_["xyz"])[0:2]
                im_x_pix, im_y_pix = coords_to_pixel(point, 
                                                     map_origin=d["map_origin"], 
                                                     map_resolution=d["map_resolution"])

                if 0 <= im_x_pix <= w and 0 <= im_y_pix <= h:
                    files.append((npz, d))

            self.scenes_data.extend(files)

        if not split_on_train_test:
            split_ratio = 1.0

        ratio = int(len(self.scenes_data) * split_ratio) 
        self.train_part = self.scenes_data[0:ratio:]
        self.test_part = self.scenes_data[ratio::]

        self.train_part = _AnyVisLocPart(self.train_part, satellite_resize, image_resize, tile_size=tile_size)
        self.test_part = _AnyVisLocPart(self.test_part, satellite_resize, image_resize, tile_size=tile_size)


    def get_train_part(self) -> _AnyVisLocPart:
        return self.train_part


    def get_test_part(self) -> _AnyVisLocPart:
        return self.test_part