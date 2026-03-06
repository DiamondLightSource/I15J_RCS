from src.main import process_image, annotate_image
from PIL import Image
from pathlib import Path
import numpy as np
import cv2


def test_given_bad_image_expected_number_of_locations_found():
    test_file_path = Path(__file__).parent / "test_data" / "bad_image.jpg"
    coordinates_path = Path(__file__).parent / "test_data" / "coordinates.json"

    image = Image.open(test_file_path)

    result, _, _ = process_image(image, coordinates_path)
    assert len(result) == 19
    assert result[1] == "Lid"
    assert result[2] == "Lid"


def test_given_bad_image_show_annotated():
    test_data_folder = Path(__file__).parent / "test_data"
    test_file_path = test_data_folder / "bad_image.jpg"
    coordinates_path = test_data_folder / "coordinates.json"
    annotated_image_path = test_data_folder / "bad_image_annotated.jpg"

    image = Image.open(test_file_path)

    result, centers, image = process_image(image, coordinates_path)
    annotated_image = annotate_image(result, centers, image)

    image_encode = cv2.imencode(".jpg", annotated_image)[1]

    with open(annotated_image_path, "rb") as f:
        disk_bytes = f.read()

    assert np.array_equal(image_encode, np.frombuffer(disk_bytes, dtype=np.uint8)), (
        "Annotated bad image not the same as expected"
    )
