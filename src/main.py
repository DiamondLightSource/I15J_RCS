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
from typing import Any, NewType, List
from fastapi.middleware.cors import CORSMiddleware
import math
import os

NParray = NewType("NParray", np.ndarray)
Responses = NewType("Responses", models.Response)
FastAPIClass = NewType("FastAPIClass", applications.FastAPI)

app = FastAPI()

origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_MOCKED_DATA = os.environ.get("USE_MOCKED_DATA", False)
CAM_URL = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"

storage = {"dewarp_coords": [], "position_centres": []}


def get_image():
    """Function that requests the image from the camera"""
    if USE_MOCKED_DATA:
        test_data_folder = Path(__file__).parent.parent / "tests" / "test_data"
        test_file_path = test_data_folder / "bad_image.jpg"
        return Image.open(test_file_path)
    response = get(CAM_URL)
    if response.status_code != 200:
        raise Exception("Camera not available")

    return Image.open(BytesIO(response.content))


def sort_anti_clockwise(points: list[list]):
    """Sort points anticlockwise, starting from the top left. Assumes 4 points."""

    pts = sorted(points, key=lambda p: p[1])

    top_points = pts[:2]
    bottom_points = pts[2:]

    top_left, top_right = sorted(top_points, key=lambda p: p[0])
    bottom_left, bottom_right = sorted(bottom_points, key=lambda p: p[0])

    return [top_left, bottom_left, bottom_right, top_right]


def dewarp_image(image: ImageFile):
    image_array = np.asarray(image)
    image_array = cv2.resize(image_array, (0, 0), fx=0.6, fy=0.6)

    width, height = image.size

    assert storage["dewarp_coords"], "Dewarp co-ordinates not set, please calibrate"

    dewarp_coords = [
        [point["x"] * width * 0.6, point["y"] * height * 0.6]
        for point in storage["dewarp_coords"]
    ]

    dewarp_coords = sort_anti_clockwise(dewarp_coords)

    dewarped_image = dewarp(image_array, dewarp_coords)
    dewarped_image_bw = cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2GRAY)
    return dewarped_image, dewarped_image_bw


def process_image(image: ImageFile):
    """Function that processes a given image and finds pucks/lids."""
    dewarped_image, dewarped_image_bw = dewarp_image(image)
    centers, result = Canning(dewarped_image_bw)
    return result, centers, dewarped_image


def annotate_image(result, centers, image):
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
    image = get_image()
    result = process_image(image)[0]
    return {"result": result}


@app.get("/position/{position}")
def position(position: int):
    """API Call for specific position state"""
    if position > 20 or position < 1:
        state: str = "Not a valid position"
    else:
        image = get_image()
        result = process_image(image)[0]
        state: str = result[position]
    return {position: state}


@app.get(
    "/annotated_image",
    responses={200: {"content": {"image/jpg": {}}}},
    response_class=Response,
)
def image():
    """API call for annotated image of the processed image"""
    image = get_image()
    result, centers, image = process_image(image)

    annotated_image = annotate_image(result, centers, image)

    image_encode = cv2.imencode(".jpg", annotated_image)[1]
    data_encode = np.array(image_encode)
    byte_encode = data_encode.tobytes()

    return Response(byte_encode, media_type="image/jpg")


@app.get(
    "/raw_image",
    responses={200: {"content": {"image/jpg": {}}}},
    response_class=Response,
)
def raw_image():
    """API call for annotated image of the processed image"""
    image = get_image()

    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    return Response(buf.getvalue(), media_type="image/jpg")


@app.get(
    "/dewarped_image",
    responses={200: {"content": {"image/jpg": {}}}},
    response_class=Response,
)
def dewarped_image():
    """API call for annotated image of the processed image"""
    image = get_image()
    image, _ = dewarp_image(image)

    image_encode = cv2.imencode(".jpg", image)[1]
    data_encode = np.array(image_encode)
    byte_encode = data_encode.tobytes()

    return Response(byte_encode, media_type="image/jpg")


@app.post("/dewarp_coordinates")
def store_dewarp_coordinates(payload: list[Any]):
    """API call to store the corners of the table for dewarping.

    The expected format is a list of dictionaries like:
        [{"x": 0.987, "y": 0.67868}, ...]

    Where the points are given as fractions of the image width/height
    """
    storage["dewarp_coords"] = payload
    return {"message": "Value stored"}


@app.get("/dewarp_coordinates")
def get_dewarp_coordinates():
    """API call to get the stored corners of the table for dewarping.

    The returned format is a list of dictionaries like:
        [{"x": 0.987, "y": 0.67868}, ...]

    Where the points are given as fractions of the image width/height
    """
    return storage["dewarp_coords"]


@app.post("/position_centres")
def store_position_centres(payload: list[Any]):
    storage["position_centres"] = payload
    return {"message": "Value stored"}


@app.get("/position_centres")
def get_position_centres():
    return storage["position_centres"]
