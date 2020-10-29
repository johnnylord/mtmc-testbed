"""Testing perspective transform (bird-view)

Demo scenario:
    Prepare a piece of paper, and write something on it. Shoot the screen with
    your camera from any angle and displacement. Select four points from the
    screen, then you will get the bird view of the selected region.
"""
import argparse
import cv2
import numpy as np

four_points = []

def mouse_callback(event, x, y, flags, param):
    """Select panel to be focused"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(four_points) < 4:
            four_points.append([x, y])
            print(four_points)
    elif event == cv2.EVENT_MOUSEMOVE:
        print("Coordinate: ({}, {})".format(x, y))

cap = cv2.VideoCapture(2)
cv2.namedWindow("Frame", cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback("Frame", mouse_callback)
cv2.namedWindow("Perspective", cv2.WINDOW_GUI_EXPANDED)

while True:
    _, frame = cap.read()

    # Draw selected points
    for pt in four_points:
        pt = tuple([ int(v) for v in pt ])
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    # Show perspective transformed result
    if len(four_points) >= 4:
        pts1 = np.array(four_points, dtype=np.float32)
        pts2 = np.array(four_points, dtype=np.float32)
        xmin, ymin = np.min(pts2[:, 0]), np.min(pts2[:, 1])
        xmax, ymax = np.max(pts2[:, 0]), np.max(pts2[:, 1])
        pts2 = np.array([[0, 0],
                        [xmax-xmin, 0],
                        [0, ymax-ymin],
                        [xmax-xmin, ymax-ymin]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (int(xmax-xmin), int(ymax-ymin)))
        cv2.imshow("Perspective", result)

    # Keyhandler
    key = cv2.waitKey(1)
    if (
        key == ord('q')
        or key == 27
    ):
        break
    elif key == ord('r'):
        four_points = []

cap.release()
cv2.destroyAllWindows()
