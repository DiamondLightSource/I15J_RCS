import os
import cv2
from Canny_Edge import Canning, dewarp
import json
from PIL import Image
from requests import get, models
from io import BytesIO
import numpy as np
from fastapi import FastAPI, Response, applications
from typing import NewType, List

NParray = NewType("NParray", np.ndarray)
Responses = NewType("Responses", models.Responses)
FastAPIClass = NewType("FastAPIClass", applications.FastAPI)

app: FastAPIClass = FastAPI()


def processing():
    """Function that requests the image from the camera and applies"""
    # Call image from live camera feed
    url: str = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
    response: Responses = get(url)
    if response.status_code != 200:
        print("Camera not available")
    else:
        image: NParray = Image.open(BytesIO(response.content))
        image: NParray = np.asarray(image)
        image: NParray = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)

    with open("/workspace/coordinates.json", "r") as file:
        input_coordinates: List[List[int]] = json.load(file)

    # # Call from images
    # dir_path = "/workspace/data/task_lights_on"
    # count = 0
    # # Iterate directory
    # for path in os.listdir(dir_path):
    #     # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         count += 1

    # for i in range(count):
    #     # this part of the code will be replaced with collecting images from the stream.
    #     # image = cv2.imread(f"{dir_path}/{i+1}.jpg", 0)
    #     image = cv2.imread(f"{dir_path}/{i+1}.jpg", 1)
    #     image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)

    dewarped: NParray = dewarp(image, input_coordinates)
    dewarped_bw: NParray = cv2.cvtColor(dewarped, cv2.COLOR_BGR2GRAY)
    centers, result = Canning(dewarped_bw)
    return result, centers, dewarped


@app.get("/result")
def result():
    """API Call for all position states"""
    result = processing()[0]
    return {"result": result}


@app.get("/position/{position}")
def position(position: int):
    """API Call for specific position state"""
    if position > 20 or position < 1:
        state: str = "Not a valid position"
    else:
        result = processing()[0]
        state: str = result[position]
    return {position: state}


@app.get(
    "/image", responses={200: {"content": {"image/jpg": {}}}}, response_class=Response
)
def image():
    """API call for annotated image of the processed image"""
    result, centers, image = processing()
    # Annotate Image based off of processing result
    for pos, circle_param in zip(centers.keys(), centers.values()):
        x, y = circle_param[0:2]
        match result[pos]:
            case "None":
                pass
            case "Puck":
                cv2.circle(image, (x, y), 130, (0, 255, 0), thickness=4)
            case "Lid":
                cv2.drawMarker(
                    image,
                    (x, y),
                    color=[0, 0, 255],
                    thickness=4,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    line_type=cv2.LINE_AA,
                    markerSize=150,
                )

    image_encode = cv2.imencode(".jpg", image)[1]
    data_encode = np.array(image_encode)
    byte_encode = data_encode.tobytes()

    return Response(byte_encode, media_type="image/jpg")
