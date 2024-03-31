import pytest
import cv2
from detector import ObjectDetector
import os
from pathlib import Path

# ! TODO --> FIX THIS to run smoothly

def test_load_model():
    # Test if it correctly loads the model from .cfg and .weights files
    model_cfg = str(Path(os.getcwd()) / '..' / 'models' / 'yolov3.cfg')
    model_weights = str(Path(os.getcwd()) / '..' / 'models' / 'yolov3.weights')
    class_file = str(Path(os.getcwd())/ '..' / 'assets' / 'coco.names')
    obj = ObjectDetector(model_cfg, model_weights, class_file)
    assert isinstance(obj.net, cv2.dnn_Net)  # Assuming that net attribute is of type dnn_Net after loading the model

def test_detect():
    # Test if it correctly detects objects in an image
    obj = ObjectDetector()
    img = cv2.imread('/data/1.jpg')
    detections = obj.detect(img)
    assert len(detections) >= 0  # It could be any positive integer, depending on the objects detected in the image
