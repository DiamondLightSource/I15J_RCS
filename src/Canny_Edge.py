import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, NewType
NParray = NewType("NParray", np.ndarray)

# Resize Image

def rescaleFrame(frame: NParray, scale: float=0.60) -> NParray:
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def dewarp(image: NParray, inputs, plot=False) -> NParray:
    assert image is not None, "file could not be read, check with os.path.exists()"
    rows, cols = image.shape
    input_points = np.float32([inputs[0], inputs[1], inputs[2], inputs[3]])
    output_points = np.float32([[0, 0], [0, rows], [rows, rows], [rows, 0]])
    M = cv2.getPerspectiveTransform(input_points, output_points)
    dst = cv2.warpPerspective(image, M, (rows, rows))

    if plot == True:
        cv2.imshow("originial", image)
        cv2.imshow("dewarped", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dst


def thresholding(image: NParray, plot: bool =False) -> dict:
    thresh: Dict[str,NParray] = {}
    for i in range(50, 121, 5):
        ret, thresh[str(i)] = cv2.threshold(image, i, 0, cv2.THRESH_TOZERO)

    # Put in Documentation: Thresholding - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html

    if plot == True:
        # Plotting
        titles = list(thresh.keys())
        images = list(thresh.values())
        length = (len(titles) // 2) + 1
        for i in range(len(titles)):
            plt.subplot(2, length, i + 1), plt.imshow(
                images[i], "gray", vmin=0, vmax=255
            )
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    return thresh


def adding_circles(circles, image,plot=False):
    for pt in circles[0, :]:
        x, y, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (x, y), 5, (0, 0, 255), 3)

    if plot == True:
        cv2.imshow("circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def Canning(image: NParray) -> NParray:
    x = thresholding(image)
    a50_edges_canny = cv2.Canny(x["50"], 150, 250)
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

        # Draw the circumference of the circle.
        image_drawn = adding_circles(detected_circles, image)
        
        
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
    counter = 1
    lids = 0
    pucks = 0
    none = 0
    for i in range(len(centers)):
        for j in centers[i]:
            centers_sorted[counter] = j
            counter += 1

    for circle_param in centers_sorted.values():
        if circle_param[2] < 100:
            none += 1
        else:
            # Mask Image with Circle
            mask = np.zeros_like(image)
            mask = cv2.circle(
                mask,
                (circle_param[0], circle_param[1]),
                circle_param[2],
                (255, 255, 255),
                -1,
            )
            masked = cv2.bitwise_and(image, image, mask=mask)

            # Find small circles
            mini_circles = cv2.HoughCircles(
                masked,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=25,
                param1=70,
                param2=22,
                minRadius=7,
                maxRadius=25,
            )

            if mini_circles is not None:
                mini_circles = np.uint16(np.around(mini_circles))
                # show_circles = adding_circles(mini_circles, masked, plot=True)
                pucks += 1
            else:
                lids += 1
            pass
    print(f"Lids: {lids}, Pucks: {pucks}, None: {none}")

    # # show Canny Edge Detection
    # cv2.imshow("edges", a50_edges_canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_drawn
