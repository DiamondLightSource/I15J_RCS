import os
import cv2
from Canny_Edge import Canning, dewarp

def main():
    # folder path: Code that is used to collect images. This will change to the camera stream
    dir_path = "data/task_lights_on"
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1

    input_coordinates = [[120, 24], [150, 1202], [1332, 1211], [1330, 0]]
    # call this from JSON.

    for i in range(count):
        # this part of the code will be replaced with collecting images from the stream.
        # image = cv2.imread(f"{dir_path}/{i+1}.jpg", 0)
        image = cv2.imread("nothing.jpg", 0)
        image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
        dewarped = dewarp(image, input_coordinates)
        output = Canning(dewarped)
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
