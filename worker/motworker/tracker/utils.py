import numpy as np

def xyah_to_tlbr(xyah):
    # Type checking
    if isinstance(xyah, np.ndarray):
        xyah = xyah.tolist()
    elif isinstance(xyah, list) or isinstance(xyah, tuple):
        xyah = xyah
    else:
        raise Exception("Cannot handle data of type {}".format(type(xyah)))

    # Conversion
    cx, cy, a, h = tuple(xyah)
    tl_x, tl_y = cx-(a*h/2), cy-(h/2)
    br_x, br_y = cx+(a*h/2), cy+(h/2)

    return tl_x, tl_y, br_x, br_y

def xyah_to_tlwh(xyah):
    # Type checking
    if isinstance(xyah, np.ndarray):
        xyah = xyah.tolist()
    elif isinstance(xyah, list) or isinstance(xyah, tuple):
        xyah = xyah
    else:
        raise Exception("Cannot handle data of type {}".format(type(xyah)))

    # Conversion
    cx, cy, a, h = tuple(xyah)
    tl_x, tl_y = cx-(a*h/2), cy-(h/2)
    w, h = a*h, h

    return tl_x, tl_y, w, h

def tlbr_to_xyah(tlbr):
    # Type checking
    if isinstance(tlbr, np.ndarray):
        tlbr = tlbr.tolist()
    elif isinstance(tlbr, list) or isinstance(tlbr, tuple):
        tlbr = tlbr
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlbr)))

    # Conversion
    tl_x, tl_y, br_x, br_y = tuple(tlbr)
    cx, cy = (tl_x+br_x)/2, (tl_y+br_y)/2
    a, h = (br_x-tl_x)/(br_y-tl_y), (br_y-tl_y)

    return cx, cy, a, h

def tlbr_to_tlwh(tlbr):
    # Type checking
    if isinstance(tlbr, np.ndarray):
        tlbr = tlbr.tolist()
    elif isinstance(tlbr, list) or isinstance(tlbr, tuple):
        tlbr = tlbr
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlbr)))

    # Conversion
    tl_x, tl_y, br_x, br_y = tuple(tlbr)
    w, h = (br_x-tl_x), (br_y-tl_y)

    return tl_x, tl_y, w, h

def tlwh_to_tlbr(tlwh):
    # Type checking
    if isinstance(tlwh, np.ndarray):
        tlwh = tlwh.tolist()
    elif isinstance(tlwh, list) or isinstance(tlwh, tuple):
        tlwh = tlwh
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlwh)))

    # Conversion
    tl_x, tl_y, w, h = tuple(tlwh)
    return tl_x, tl_y, tl_x+w, tl_y+h

def tlwh_to_xyah(tlwh):
    # Type checking
    if isinstance(tlwh, np.ndarray):
        tlwh = tlwh.tolist()
    elif isinstance(tlwh, list) or isinstance(tlwh, tuple):
        tlwh = tlwh
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlwh)))

    # Conversion
    tl_x, tl_y, w, h = tuple(tlwh)
    cx, cy = tl_x+(w/2), tl_y+(h/2)
    a = w/h
    return cx, cy, a, h

def compute_iou(box_a, box_b):
    # Type checking
    if isinstance(box_a, np.ndarray):
        box_a = box_a.tolist()
    if isinstance(box_b, np.ndarray):
        box_b = box_b.tolist()

    # determine the (x, y) coordinate of the intersection rectangle
    inter_tl_x = max(box_a[0], box_b[0])
    inter_tl_y = max(box_a[1], box_b[1])
    inter_br_x = min(box_a[2], box_b[2])
    inter_br_y = min(box_a[3], box_b[3])
    box_inter = [inter_tl_x, inter_tl_y, inter_br_x, inter_br_y]

    # Copmute the areas
    area_inter = max(0, box_inter[2]-box_inter[0]) * max(0, box_inter[3]-box_inter[1])
    area_a = max(0, box_a[2]-box_a[0]) * max(0, box_a[3]-box_a[1])
    area_b = max(0, box_b[2]-box_b[0]) * max(0, box_b[3]-box_b[1])

    iou = area_inter / float(area_a + area_b - area_inter)
    return iou
