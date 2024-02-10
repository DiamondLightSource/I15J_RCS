import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, NewType

NParray = NewType("NParray", np.ndarray)


def dewarp(image: NParray, inputs, plot=False) -> NParray:
    assert image is not None, "file could not be read, check with os.path.exists()"
    rows, cols, depth = image.shape
    input_points = np.float32([inputs[0], inputs[1], inputs[2], inputs[3]])
    output_points = np.float32([[0, 0], [0, rows], [rows, rows], [rows, 0]])
    M = cv2.getPerspectiveTransform(input_points, output_points)
    dst = cv2.warpPerspective(image, M, (rows, rows))

    if plot:
        cv2.imshow("originial", image)
        cv2.imshow("dewarped", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dst


def thresholding(image: NParray, plot: bool = False) -> dict:
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5
    )
    # Not used, but a good alternative
    thresh1 = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3.5
    )

    if plot:
        # Plotting
        cv2.imshow("Gaussian Adaptive Threshold", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return thresh


def fourier(image: NParray, plot: bool = False):
    fourier = np.fft.fft2(np.float32(image))
    fourier_shift = np.fft.fftshift(abs(fourier))
    if plot:
        fig, axs = plt.subplots(2)
        axs[0].imshow(image, cmap="grey")
        top = np.quantile(fourier_shift, 0.98)
        bottom = np.quantile(fourier_shift, 0.2)
        axs[1].imshow(fourier_shift, vmin=bottom, vmax=top)
        plt.show()
        pass


def template(image: NParray):
    template = cv2.imread("templates/lid.jpg", 0)
    h, w = template.shape

    methods = [
        cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
    ]

    for method in methods:
        img2 = image.copy()

        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0] + w, location[1] + h)
        cv2.rectangle(img2, location, bottom_right, 255, 5)
        cv2.imshow("Match", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pass


def adding_circles(circles, image, plot=False):
    for pt in circles[0, :]:
        x, y, r = pt[0], pt[1], pt[2]
        # Draw the circumference of the circle.
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        # cv2.circle(image, (x, y), 5, (0, 0, 255), 3)

    if plot:
        cv2.imshow("circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def Canning(image: NParray) -> NParray:
    thresh = thresholding(image)
    a50_edges_canny = cv2.Canny(thresh, 150, 250)
    a50_edges_gauss = cv2.GaussianBlur(thresh, (5, 5), 0)

    detected_circles = cv2.HoughCircles(
        a50_edges_gauss,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=215,
        param1=260,
        param2=42,
        minRadius=85,
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
    result: Dict[int, str] = {}
    counter: int = 1

    for i in range(len(centers)):
        for j in centers[i]:
            centers_sorted[counter] = j
            counter += 1

    for circle_param, pos in zip(centers_sorted.values(), centers_sorted.keys()):
        if circle_param[2] < 100:
            result[pos] = "None"
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

            # slicing to make r = 130 (aka max radiums)
            y, x, r = circle_param
            if r > x:
                x_lower = 0
            else:
                x_lower = x - r
            if r > y:
                y_lower = 0
            else:
                y_lower = y - r

            # Slicing image into sections for quicker analysis
            masked2 = masked[x_lower : x + r, y_lower : y + r]
            # fourier(masked2)
            # template(masked2)

            # Find small circles
            mini_circles = cv2.HoughCircles(
                masked2,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=25,
                param1=50,
                param2=20,
                minRadius=7,
                maxRadius=25,
            )

            if mini_circles is not None:
                mini_circles = np.uint16(np.around(mini_circles))
                if np.shape(mini_circles)[1] > 5:
                    show_circles = adding_circles(mini_circles, masked2)
                    result[pos] = "Puck"
                else:
                    result[pos] = "Lid"
            else:
                result[pos] = "Lid"
            pass

    # show Canny Edge Detection
    # cv2.imshow("edges", a50_edges_canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return centers_sorted, result
