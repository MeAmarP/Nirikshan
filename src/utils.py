import cv2
import numpy as np

class Color:
    """
    Class for color constants.
    """
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    purple = (128, 0, 128)
    brown = (165, 42, 42)
    gray = (128, 128, 128)
    pink = (255, 192, 203)
    teal = (128, 128, 0)

def display_detections(frame: np.array, detections):
    """_summary_

    Args:
        frame (np.array): _description_
        detections (_type_): _description_
    """
    for dets in detections:
        bbox = dets['box']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), Color.green, 2)
        cv2.putText(frame, dets['class_name'], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, Color.orange, 2)

def display_tracked_ids(frame: np.array, tracked_objects):
    """_summary_

    Args:
        frame (np.array): _description_
        tracked_objects (_type_): _description_
    """
    for obj in tracked_objects:
        bbox = obj.tlwh.astype(np.int32)
        id = obj.track_id
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), Color.cyan, 2)
        cv2.putText(frame, str(id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, Color.orange, 2)

# function to display the analytics on frame
def display_analytics(frame: np.array , analytics):
    """_summary_

    Args:
        frame (np.array): _description_
        analytics_metrics (dict): _description_

    Returns:
        _type_: _description_
    """
    if analytics.name == 'CountAnalytics':
        values = analytics.get()
        for key in values:
            cv2.putText(frame, f"{key}: {values[key]}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, Color.purple, 2)
    return frame
