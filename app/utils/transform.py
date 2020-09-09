import numpy as np


def convert_bbox_coordinate(bboxes, old_resolution, new_resolution):
    """Convert bboxes' coordinates from old resolution to new resolution

    Args:
        bboxes (list): list of tlbr sequence of type list
        old_resolution (tuple): old screen size (width, height)
        new_resolution (tuple): new screen size (width, height)

    Returns:
        a list of tlbr sequence in new coordinate system
    """
    if len(bboxes) == 0:
        return []

    ratios = np.array(new_resolution) / np.array(old_resolution)
    x_ratio, y_ratio = ratios[0], ratios[1]

    bboxes = np.array(bboxes)
    bboxes[:, 0] = (bboxes[:, 0]*x_ratio).astype(np.int)
    bboxes[:, 2] = (bboxes[:, 2]*x_ratio).astype(np.int)
    bboxes[:, 1] = (bboxes[:, 1]*y_ratio).astype(np.int)
    bboxes[:, 3] = (bboxes[:, 3]*y_ratio).astype(np.int)
    bboxes = bboxes.tolist()
    return bboxes
