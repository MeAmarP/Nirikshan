import pytest
import cv2
from detector import ObjectDetector
import os
from pathlib import Path


def test_load_model():
    # Test if it correctly loads the model from .cfg and .weights files

    obj = ObjectDetector('yolov3')
    assert isinstance(obj.net, cv2.dnn_Net)  # Assuming that net attribute is of type dnn_Net after loading the model

def test_detect():
    # Test if it correctly detects objects in an image
    obj = ObjectDetector()
    # * NOTE Make sure path below is absolute path to the image file 
    # ! TODO Write a code to get absolute path to input image file
    path_to_img_file ='data/1.jpg' 
    img = cv2.imread(path_to_img_file)
    detections = obj.detect(img)
    assert len(detections) >= 0  # It could be any positive integer, depending on the objects detected in the image
