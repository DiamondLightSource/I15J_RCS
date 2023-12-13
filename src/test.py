import numpy as np
import cv2
import cProfile, pstats, io
from Canny_Edge import dewarp
from imutils.object_detection import non_max_suppression
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()

input_coordinates = [[120, 24], [150, 1202], [1332, 1211], [1330, 0]]
img = cv2.resize(cv2.imread("data/task_lights_on/5.jpg", 0), (0, 0), fx=0.6, fy=0.6)
img = dewarp(img, input_coordinates)
template = cv2.imread("templates/puck.jpg", 0)
h, w = template.shape
threshold = 0.96

methods = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED,
]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    # (yCoords, xCoords) = np.where(result >= threshold)

    # rects = []

    # for x, y in zip(xCoords, yCoords):
    #     cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #     rects.append((x, y, x + w, y + h))

    # pick = non_max_suppression(np.array(rects))
    # for startX, startY, endX, endY in pick:
    #     cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    location = np.where(result >= threshold)
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

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


# # skimage coin with varying background

# # # https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
# # contours, hierarchy = cv2.findContours(a["50"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# # for contour in contours:
# #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
# #     cv2.drawContours(a["50"], [approx], 0, (0, 0, 0), 5)
# #     x = approx.ravel()[0]
# #     y = approx.ravel()[1] - 5


# # if len(approx) == 3:
# #     cv2.putText(a["50"], "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
# # elif len(approx) == 4:
# #     x, y, w, h = cv2.boundingRect(approx)
# #     aspectRatio = float(w) / h
# #     print(aspectRatio)
# #     if aspectRatio >= 0.95 and aspectRatio < 1.05:
# #         cv2.putText(a["50"], "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

# #     else:
# #         cv2.putText(
# #             a["50"], "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
# #         )
# ____________________________________________
# import cv2
# import numpy as np

# # https://stackoverflow.com/questions/64491530/how-to-remove-the-background-from-a-picture-in-opencv-python

# # Read image
# img = cv2.imread("data/task_lights_off/20.jpg")
# hh, ww = img.shape[:2]

# # threshold on white
# # Define lower and uppper limits
# lower = np.array([100, 100, 100])
# upper = np.array([255, 255, 255])

# # Create mask to only select black
# thresh = cv2.inRange(img, lower, upper)

# # apply morphology
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # invert morp image
# mask = 255 - morph

# # apply mask to image
# result = cv2.bitwise_and(img, img, mask=mask)


# # save results
# # cv2.imwrite("/results/on_thresh.jpg", thresh)
# # cv2.imwrite("/results/on_morph.jpg", morph)
# # cv2.imwrite("/results/on_mask.jpg", mask)
# # cv2.imwrite("/results/on_result.jpg", result)

# cv2.imshow("thresh", thresh)
# cv2.imshow("morph", morph)
# cv2.imshow("mask", mask)
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
