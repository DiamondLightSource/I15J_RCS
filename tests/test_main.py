from src.main import (
    process_image,
    annotate_image,
    sort_anti_clockwise,
    store_dewarp_coordinates,
)
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import pytest
import json


def test_given_bad_image_expected_number_of_locations_found():
    test_file_path = Path(__file__).parent / "test_data" / "bad_image.jpg"
    coordinates_path = Path(__file__).parent / "test_data" / "coordinates.json"

    with open(coordinates_path) as file:
        input_coordinates = json.load(file)

    store_dewarp_coordinates(input_coordinates)

    image = Image.open(test_file_path)

    result, _, _ = process_image(image)
    assert len(result) == 19
    assert result[1] == "Lid"
    assert result[2] == "Lid"


def test_given_bad_image_show_annotated():
    test_data_folder = Path(__file__).parent / "test_data"
    test_file_path = test_data_folder / "bad_image.jpg"
    coordinates_path = test_data_folder / "coordinates.json"
    annotated_image_path = test_data_folder / "bad_image_annotated.jpg"

    image = Image.open(test_file_path)

    with open(coordinates_path) as file:
        input_coordinates = json.load(file)

    store_dewarp_coordinates(input_coordinates)

    result, centers, image = process_image(image)
    annotated_image = annotate_image(result, centers, image)

    image_encode = cv2.imencode(".jpg", annotated_image)[1]

    with open(annotated_image_path, "rb") as f:
        disk_bytes = f.read()

    assert np.array_equal(image_encode, np.frombuffer(disk_bytes, dtype=np.uint8)), (
        "Annotated bad image not the same as expected"
    )


def test_given_trapezoid_with_higher_top_right_then_sorted_correctly():
    pts = [
        [100, 50],  # TL
        [400, 40],  # TR (slightly higher)
        [420, 300],  # BR
        [120, 310],  # BL
    ]

    tl, bl, br, tr = sort_anti_clockwise(pts)

    assert tl == [100, 50]
    assert bl == [120, 310]
    assert br == [420, 300]
    assert tr == [400, 40]


def test_given_trapezoid_in_random_order_then_sorted_correctly():
    pts = [
        [420, 300],  # BR
        [100, 50],  # TL
        [120, 310],  # BL
        [400, 40],  # TR
    ]

    tl, bl, br, tr = sort_anti_clockwise(pts)

    assert tl == [100, 50]
    assert bl == [120, 310]
    assert br == [420, 300]
    assert tr == [400, 40]


def test_given_trapezoid_with_nearly_equal_y_values_then_sorted_correctly():
    pts = [
        [100, 50.0001],  # TL
        [400, 50.0000],  # TR (slightly higher)
        [420, 300],  # BR
        [120, 310],  # BL
    ]

    tl, bl, br, tr = sort_anti_clockwise(pts)

    assert tl == [100, 50.0001]
    assert tr == [400, 50.0000]


def test_given_shape_with_too_many_points_then_sorting_errors():
    pts = [
        [100, 50],
        [400, 50],
        [420, 300],
        [120, 310],
        [50, 96],
    ]

    with pytest.raises(ValueError):
        sort_anti_clockwise(pts)


def test_given_shape_with_too_few_points_then_sorting_errors():
    pts = [
        [100, 50],
        [400, 50],
    ]

    with pytest.raises(ValueError):
        sort_anti_clockwise(pts)
