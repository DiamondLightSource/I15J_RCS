import cv2
from Canny_Edge import thresholding
from typing import List, Dict
import numpy as np

a: list[list[int]] = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pointer at: {x},{y}")
        a.append([x, y])


def calibration():
    """Function for the user to choose 4 differnt points in the code that will allow the"""

    dir_path = "data/task_lights_on"
    image = cv2.imread(f"{dir_path}/1.jpg", 0)
    image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
    cv2.namedWindow("Point Coordinates")
    y = cv2.setMouseCallback("Point Coordinates", click_event)

    while True:
        cv2.imshow("Point Coordinates", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()

    x = thresholding(image)
    a50_edges_gauss = cv2.GaussianBlur(x["50"], (5, 5), 0)

    detected_circles = cv2.HoughCircles(
        a50_edges_gauss,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=210,
        param1=250,
        param2=25,
        minRadius=75,
        maxRadius=130,
    )

    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        centers: List[List[int]] = []
        for pt in detected_circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            centers.append([x, y, r])

    # Sorting Centers of circles
    centers = sorted(centers, key=lambda x: x[1])
    centers = [
        sorted(centers[0:3], key=lambda x: x[0]),
        sorted(centers[3:6], key=lambda x: x[0]),
        sorted(centers[6:11], key=lambda x: x[0]),
        sorted(centers[11:16], key=lambda x: x[0]),
        sorted(centers[16:], key=lambda x: x[0]),
    ]
    centers_sorted: Dict[int, List[int]] = {}

    counter: int = 1
    for i in range(len(centers)):
        for j in centers[i]:
            centers_sorted[counter] = j
            counter += 1

    # a = input coordinates, centers
    # OUTPUT AS A JSON (pythoninbuilt module json)
    # https://docs.python.org/3/library/json.html
    # JSON dumps and JSON loads
    # Exports as filename
    return a, centers_sorted


calibration()
