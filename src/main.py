import os
import cv2
from Canny_Edge import Canning, dewarp
import json
from PIL import Image
from requests import get
from io import BytesIO
import numpy as np
from fastapi import FastAPI

app = FastAPI()

def main():
    # Call image from live camera feed
    url = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
    response = get(url)
    if response.status_code != 200:
        print("Camera not available")
    else:
        image = Image.open(BytesIO(response.content))
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(0, 0), fx=0.6, fy=0.6)

    with open("coordinates.json","r") as file:
        input_coordinates = json.load(file)


    # # Call from images
    # dir_path = "data/task_lights_on"
    # count = 0
    # # Iterate directory
    # for path in os.listdir(dir_path):
    #     # check if current path is a file
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         count += 1

    # for i in range(count):
    #     # this part of the code will be replaced with collecting images from the stream.
    #     image = cv2.imread(f"{dir_path}/{i+1}.jpg", 0)
    #     image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)

        
    dewarped = dewarp(image, input_coordinates)
    output = Canning(dewarped)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
