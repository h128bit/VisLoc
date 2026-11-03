import os
import rasterio 
import numpy as np
from pathlib import Path


def conver_image_to_tiff(path_to_png: str|os.PathLike|Path, 
                         path_to_save_tiff: str|os.PathLike|Path, 
                         tile_size:int|None=None):
    titling = tile_size is not None

    with rasterio.open(path_to_png) as png_file:
        profile = png_file.profile.copy()
        profile.update(
            driver='GTiff',
            tiled=titling,
            blockxsize=tile_size,
            blockysize=tile_size,
            compress='lzw',
            bigtiff='yes',
            interleave='pixel',
        )

        with rasterio.open(path_to_save_tiff, 'w', **profile) as dst:
            dst.write(png_file.read())


def load_tile(path_to_tiff_file: str|os.PathLike|Path, tile_idx: tuple[int, int]):
    with rasterio.open(path_to_tiff_file) as src:
        block_h, block_w = src.block_shapes[0]

        col_idx, row_idx = tile_idx
        
        window = rasterio.windows.Window(
            col_off=col_idx * block_w,
            row_off=row_idx * block_h,
            width=block_w,
            height=block_h,
        )

        tile = src.read(indexes=[1, 2, 3], window=window).astype(np.uint8)
        tile = tile.transpose([1, 2, 0])

    return tile