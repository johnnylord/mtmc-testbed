import cv2
import numpy as np


def draw_bbox(frame, bbox, color=(85,135,255), thickness=2):
    """Draw bounding box on the specified frame

    Args:
        frame (ndarray): processing frame
        bbox (list): tlbr sequence of type list
        color (tuple): BGR color palette
        thickness (int): line thickness
    """
    tl_x, tl_y = tuple([ int(v) for v in bbox[:2]])
    br_x, br_y = tuple([ int(v) for v in bbox[2:]])
    cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), color, thickness)
