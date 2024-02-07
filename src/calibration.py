import cv2
from Canny_Edge import thresholding,dewarp,adding_circles
from typing import List, Dict
import numpy as np
import json
from PIL import Image
from requests import get
from io import BytesIO

dewarp_coord: list[list[int]] = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pointer at: {x},{y}")
        dewarp_coord.append([x, y])


def calibration():
    """Function for the user to choose 4 differnt points in the code that will allow the"""

    url = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
    response = get(url)
    if response.status_code != 200:
        print("Camera not available")
    else:
        image = Image.open(BytesIO(response.content))
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(0, 0), fx=0.6, fy=0.6)
    cv2.namedWindow("Point Coordinates")
    y = cv2.setMouseCallback("Point Coordinates", click_event)

    while True:
        cv2.imshow("Point Coordinates", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()

    while True:
        plot = input("Would you like to see the dewarped image? Y/N ").upper()
        if plot == "Y":
            image_dewarp = dewarp(image,dewarp_coord)
            cv2.imshow("Dewarped", image_dewarp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    
    with open("coordinates.json","w") as file:
        json.dump(dewarp_coord,file)


calibration()
