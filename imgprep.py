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
                json = json.load(f)
    return json


def load_training(dir, json):
    pieces = []
    for piece in json["annotations"]:
        filename = json["images"][piece["id"]]["file_name"]
        img = cv.imread(os.path.join(dir, filename))
        bb = piece["bbox"]
        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[0] + bb[2]
        y2 = bb[1] + bb[3]
        crop = img[x1:x2, y1:y2]
        pieces.append(crop, piece["category_id"])
    return pieces



def order_points(pts):
    
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left
    
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
      
    img = cv.imread(image)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
   

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    
    img = cv.fromarray(warped, "RGB")
    # img.show()    
    # return the warped image
    return img
