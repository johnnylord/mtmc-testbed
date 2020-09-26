import cv2
import numpy as np
import colorsys
from matplotlib import cm

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

def draw_text(frame, text, position,
            fgcolor=(85, 125, 255),
            bgcolor=(85, 135, 255),
            fontScale=1, thickness=2, margin=5):
    """Draw text on the specified frame

    Args:
        frame (ndarray): processing frame
        text (string): text to render
        position (tuple): text position (tl_x, tl_y)
        fgcolor (tuple): BGR color palette for font color
        bgcolor (tuple): BGR color palette for background color
        fontScale (int): font scale
        thickness (int): line thickness
        margin (int): space between texts
    """
    # opencv doesn't handle `\n` in the text
    # therefore we handle it line by line
    lines = text.split('\n')
    text_widths = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][0]
                    for line in lines ]
    text_heights = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][1]
                    for line in lines ]
    max_width = int(max(text_widths))
    max_height = int(max(text_heights))
    tl_x = int(position[0])
    tl_y = int(position[1])

    # draw background
    cv2.rectangle(frame,
                (tl_x, tl_y),
                (tl_x+max_width, tl_y+max_height*len(lines)),
                bgcolor, -1)

    # draw text line by line
    for j, line in enumerate(lines):
        cv2.putText(frame, line,
                (tl_x+margin, tl_y+(max_height*(j+1))-margin),
                color=fgcolor,
                fontScale=fontScale,
                thickness=thickness,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

def draw_velocity(frame, position, vector, thickness):
    vl = frame.shape[1] // 100
    pos1 = tuple([ int(v) for v in position])
    pos2 = tuple([ int(v) for v in position+vector*vl ])
    if vector[0] < 0:
        degree = 360 - np.degrees(np.arctan2(vector[0], vector[1]))
    else:
        degree = np.degrees(np.arctan2(vector[0], vector[1]))

    color = (np.array(cm.jet(degree/360))*255)[:3]
    cv2.arrowedLine(frame, pos1, pos2, color=color, thickness=thickness)

def draw_gaussian(frame, mean, covariance, color, thickness):
    vals, vecs = np.linalg.eigh(5.9915 * covariance)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    center = int(mean[0] + .5), int(mean[1] + .5)
    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness)

def get_unique_color(tag, hue_step=0.41):
    h, v = (tag*hue_step) % 1, 1. - (int(tag*hue_step)%4)/5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(r*255), int(255*g), int(255*b)
