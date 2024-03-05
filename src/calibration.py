import cv2
from Canny_Edge import dewarp
from typing import List, NewType
import numpy as np
import json
from PIL import Image
from requests import get, models
from io import BytesIO


NParray = NewType("NParray", np.ndarray)
Response = NewType("Response", models.Response)
dewarp_coord: List[List[int]] = []


def click_event(event, x: int, y: int, flags, params):
    """Appends coordinate positions on the image to list on click."""
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pointer at: {x},{y}")
        dewarp_coord.append([x, y])


def calibration() -> None:
    """Function to choose 4 different points in the collected stream image that will be used to dewarp the image. This is independent of the FastAPI and can be incorporated as a service as a future update"""

    url: str = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
    response: Response = get(url)
    if response.status_code != 200:
        print("Camera not available")
    else:
        image: NParray = Image.open(BytesIO(response.content))
        image: NParray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        image: NParray = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
    cv2.namedWindow("Point Coordinates")
    y: any = cv2.setMouseCallback("Point Coordinates", click_event)

    while True:
        # Press q to close the function
        cv2.imshow("Point Coordinates", image)
        k: int = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()

    while True:
        plot = input("Would you like to see the dewarped image? Y/N ").upper()
        if plot == "Y":
            image_dewarp: NParray = dewarp(image, dewarp_coord)
            cv2.imshow("Dewarped", image_dewarp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        break

    while True:
        save = input("Would you like to save the coordinates? Y/N ").upper()
        if save == "Y":
            with open("coordinates.json", "w") as file:
                json.dump(dewarp_coord, file)
        break


calibration()
