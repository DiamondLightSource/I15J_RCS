import cv2
from matplotlib import pyplot as plt

img_on = cv2.imread("data/task_lights_on/20.jpg", 0)
img_off = cv2.imread("data/task_lights_off/20.jpg", 0)

# Collect Images
cv2.imshow("Lights on", img_on)
cv2.imshow("Greyscale", img_off)
cv2.waitKey(0)
cv2.destroyAllWindows()

a: dict = {}
for i in range(50, 121, 5):
    ret, a[str(i)] = cv2.threshold(img_on, i, 0, cv2.THRESH_TOZERO)

# Thresholding - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html

# Plotting
titles = list(a.keys())
images = list(a.values())
length = (len(titles) // 2) + 1
for i in range(len(titles)):
    plt.subplot(2, length, i + 1), plt.imshow(images[i], "gray", vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


# skimage coin with varying background

# 1
contours, hierarchy = cv2.findContours(a["50"], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(a["50"], [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5


if len(approx) == 3:
    cv2.putText(a["50"], "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
elif len(approx) == 4:
    x, y, w, h = cv2.boundingRect(approx)
    aspectRatio = float(w) / h
    print(aspectRatio)
    if aspectRatio >= 0.95 and aspectRatio < 1.05:
        cv2.putText(a["50"], "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    else:
        cv2.putText(
            a["50"], "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
        )


cv2.imshow("shapes", a["50"])
cv2.waitKey(0)
cv2.destroyAllWindows()
