from pathlib import Path
import cv2
from edge_detection import Canning, dewarp
import json
from PIL import Image
from PIL.ImageFile import ImageFile
from requests import get, models
from io import BytesIO
import numpy as np
from fastapi import FastAPI, Response, applications
from typing import NewType, List

NParray = NewType("NParray", np.ndarray)
Responses = NewType("Responses", models.Response)
FastAPIClass = NewType("FastAPIClass", applications.FastAPI)

app: FastAPIClass = FastAPI()


def process_image_from_stream():
    """Function that requests the image from the camera and processes it."""
    url = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
    response = get(url)
    if response.status_code != 200:
        print("Camera not available")
    else:
        image = Image.open(BytesIO(response.content))
        return process_image(image)


def process_image(
    image: ImageFile,
    dewarp_coordinates_path: str | Path = "/workspace/coordinates.json",
):
    """Function that processes a given image and finds pucks/lids."""
    image_array = np.asarray(image)
    image_array = cv2.resize(image_array, (0, 0), fx=0.6, fy=0.6)

    with open(dewarp_coordinates_path) as file:
        input_coordinates: List[List[int]] = json.load(file)

    dewarped: NParray = dewarp(image_array, input_coordinates)
    dewarped_bw: NParray = cv2.cvtColor(dewarped, cv2.COLOR_BGR2GRAY)
    centers, result = Canning(dewarped_bw)
    return result, centers, dewarped


def annotate_image(result, centers, image):
    # Annotate Image based off of processing result
    for pos, circle_param in zip(centers.keys(), centers.values()):
        x, y = circle_param[0:2]
        match result[pos]:
            case "None":
                cv2.circle(image, (x, y), 130, (255, 0, 0), thickness=4)
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
    return image


@app.get("/result")
def result():
    """API Call for all position states"""
    result = process_image_from_stream()[0]
    return {"result": result}


@app.get("/position/{position}")
def position(position: int):
    """API Call for specific position state"""
    if position > 20 or position < 1:
        state: str = "Not a valid position"
    else:
        result = process_image_from_stream()[0]
        state: str = result[position]
    return {position: state}


@app.get(
    "/image", responses={200: {"content": {"image/jpg": {}}}}, response_class=Response
)
def image():
    """API call for annotated image of the processed image"""
    result, centers, image = process_image_from_stream()

    annotated_image = annotate_image(result, centers, image)

    image_encode = cv2.imencode(".jpg", annotated_image)[1]
    data_encode = np.array(image_encode)
    byte_encode = data_encode.tobytes()

    return Response(byte_encode, media_type="image/jpg")
