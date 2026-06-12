import cv2
import numpy as np


def generate_mask_from_polygon(polygon: list) -> np.array:
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = polygon
    height, width = 512, 512
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array([[tl_x, tl_y],
                        [tr_x, tr_y],
                        [br_x, br_y],
                        [bl_x, bl_y]], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
    return mask