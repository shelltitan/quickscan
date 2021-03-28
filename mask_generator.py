import numpy as np
import json
import cv2
import os


json_path = "eye.json"
output_path = "D:/eyeset/videos2/grayscale/masks"
MASK_WIDTH = 480
MASK_HEIGHT = 720

with open(json_path, "r") as read_file:
    data = json.load(read_file)

for it in data["_via_img_metadata"]:
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    try:
        x_points = data["_via_img_metadata"][it]["regions"][0]["shape_attributes"]["all_points_x"]
        y_points = data["_via_img_metadata"][it]["regions"][0]["shape_attributes"]["all_points_y"]
        file_name = it
    except:
        continue
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    arr = np.array(all_points)
    cv2.fillPoly(mask, [arr], color=(255))
    cv2.imwrite(os.path.join(output_path, file_name.split(".png")[0] + ".png") , mask)
    
# TODO
# Basic GUI with OPEN JSON , OUTPUT AND IMAGE DIMENSIONS, PROGRESS BAR
# TODO
# OTHER APP to check mask on original frame
# TODO
# COMBINE THESE BOTH
# MIGHT DO
# INTEGRATE VGG AND ALL above FUNCTIONS into one APP