import os
import cv2
from Canny_Edge import Canning, dewarp, rescaleFrame


a: list[list[int]] = []
# folder path: Code that is used to
dir_path = "data/task_lights_off"

def click_event(event, x, y):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pointer at: {x},{y}")
        a.append([x, y])


#TODO: Add calibration as a separate file. This should store the input_coordinates as a file that can be read (csv?)
def calibration():
    """Function for the user to choose 4 differnt points in the code that will allow the"""
    image = cv2.imread(f"{dir_path}/1.jpg", 0)
    image = rescaleFrame(image)
    cv2.namedWindow("Point Coordinates")
    y = cv2.setMouseCallback("Point Coordinates", click_event)
    a.append(y)
    while True:
        cv2.imshow("Point Coordinates", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()

def main():
    
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1


    input_coordinates = [[120, 24], [150, 1202], [1332, 1211], [1330, 0]]

    for i in range(count):
        # this part of the code will be replaced with collecting images from the stream.
        image = cv2.imread(f"{dir_path}/{i+1}.jpg", 0)
        image = rescaleFrame(image)
        dewarped = dewarp(image, input_coordinates)
        # dewarped = cv2.equalizeHist(dewarped)
        output = Canning(dewarped)
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    