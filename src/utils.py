import cv2
import numpy as np

def display_detections(frame, detections):
    # TODO Move to new file 'utils'
    for dets in detections:
        bbox = dets['box']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, dets['class_name'], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (36,255,12), 2)

def display_tracked_ids(frame, tracked_objects):
    # TODO Move to new file 'utils'
    for obj in tracked_objects:
        bbox = obj.tlwh.astype(np.int32)
        id = obj.track_id
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
