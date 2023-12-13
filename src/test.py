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