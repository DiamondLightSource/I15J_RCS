# I15J_RCS
Beamline I15-1 / I15-J Robot Camera System aimed towards identifying the presence of sample holder lids.
This in turn prevents any attempts for samples to be collected while a lid is attached to a sample puck.
This is done using OpenCV and FastAPI to provide calls for position states, both overall ["result"] and individual ["position"] as well as an annotated image ["image"] to cross-check the accuracy of the image. 

Service Workflow: 
0. Calibrate the file to dewarp the image to collect the stage.
1. Collect Image from stream
2. Dewarp Image
3. Apply Adaptive Thresholding to the image
4. Apply Gaussian Blur to image
5. Identify and sort positions using Hough Transforms 
6. Identify state of positions ["None", "Puck", "Lid"]
7. Return FastAPI response -  ["result"], ["position"], ["image"]

Documentation:
User Guide: https://confluence.diamond.ac.uk/display/SSCC/Documentation%3A+User+Guide
Developer Guide: https://confluence.diamond.ac.uk/display/SSCC/Documentation%3A+Developer%27s+Guide