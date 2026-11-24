import cv2 as cv
import os
import sys
import numpy as np
import argparse
def Input_image(image_path=None):
    if image_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", type=str, default="./input/img.png")
        image_path=parser.parse_args().image_path
    img=cv.imread(image_path)
    return img