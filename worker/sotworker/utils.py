import cv2
import numpy as np


def crop_image(frame, bbox):
    """Return the cropped image from frame specified by bbox"""
    x_start, x_end = int(bbox[0]), int(bbox[2])
    y_start, y_end = int(bbox[1]), int(bbox[3])
    crop_img = frame[y_start:y_end, x_start:x_end, :].copy()

    return crop_img
