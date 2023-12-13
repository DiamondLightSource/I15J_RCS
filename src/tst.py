from PIL import Image
from requests import get
from io import BytesIO
import cv2
import numpy as np
import asyncio
import os
from Canny_Edge import rescaleFrame, dewarp, Canning

# import cProfile,pstats,io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()


async def main():
    count = 20
    while True:
        url = "http://bl15j-di-serv-01.diamond.ac.uk:8087/JCAM3.mjpg.jpg"
        ouput_dir = "/dls/science/groups/das/i15_1_images/raw"
        dir_proc = "/dls/science/groups/das/i15_1_images/processed"
        response = get(url)
        if response.status_code != 200:
            print("Camera not available")
            await asyncio.sleep(10)
            # continue
        else:
            img = Image.open(BytesIO(response.content))
            img = np.asarray(img)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image = rescaleFrame(gray_image)
            input_coordinates = [[120, 24], [150, 1202], [1332, 1211], [1330, 0]]
            dewarped = dewarp(gray_image, input_coordinates)
            dewarped = cv2.equalizeHist(dewarped)
            output = Canning(dewarped)
            cv2.imwrite(os.path.join(ouput_dir, f"{count}.jpg"), img)
            cv2.imwrite(os.path.join(dir_proc, f"{count}.jpg"), output)
            count += 1
            await asyncio.sleep(3600)


asyncio.run(main())
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
