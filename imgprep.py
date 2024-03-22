import cv2 as cv
import numpy as np
import os
import json


# Loads images from directory into images in format (matrix, filename) and returns a json, if found
def load_images(dir, images):
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if filename.is_file():
            if f.lower.endswith('.jpg'):
                #do something???
                images.append(cv.imread(f), f)
            elif f.lower.endswith('.json'):
                json = json.loads(f)
    return json

