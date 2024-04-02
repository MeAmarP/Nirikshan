import cv2
import numpy as np
import traceback
from pathlib import Path
import os

import json

from configs import AppConfig
from detector import ObjectDetector
from tracker.byte_tracker import BYTETracker

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

def main():
    model_cfg = Path.cwd() / AppConfig.yolov3_cfg
    model_weights = Path.cwd() / AppConfig.yolov3_weights
    class_file = Path.cwd() / AppConfig.path_coco_names

    if not (model_cfg.exists() and model_weights.exists()):
        raise FileNotFoundError('Model files not found')

    # Initialize object detector
    yolov3_detector = ObjectDetector(model_cfg=str(model_cfg), model_weights=str(model_weights), class_file=str(class_file))

    # Initialize object tracker
    tracker = BYTETracker(frame_rate=AppConfig.tracker_fps)  # Assuming 30 fps for now (can be adjusted later)

    try:
    # Open the video file
        cap = cv2.VideoCapture("/home/c3po/Documents/project/learning/amar-works/Nirikshan/data/palace.mp4")
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred while trying to open the video file: {e}")
    else:
        # If successful, continue with processing the video frames
        while True:
            try:
                # Read a frame from the video
                ret, frame = cap.read()
                
                # If the 'ret' value is False (i.e., there are no more frames to read)
                if not ret:
                    break

                detections = yolov3_detector.detect(frame, target_class_id=0) # 0 for person

                if len(detections) > 0:

                    # byte-tracker expects detection format as numpy array of shape (N, 5), 
                    # where N is the number of detections and each detection has format [x1, y1, x2, y2, confidence]  
                    np_detections = np.array([np.concatenate((np.array(det['box']).astype(np.float16), np.array([det['confidence']]).astype(np.float16))) for det in detections])
                    
                    
                    tracked_objects = tracker.update(np_detections)

                    # draw detections
                    # display_detections(frame=frame, detections=detections)

                    # draw tracked objects
                    display_tracked_ids(frame=frame, tracked_objects=tracked_objects)

                # Display the current frame
                cv2.imshow('Video', frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                traceback.print_exc()
                print(f"An error occurred while trying to process the video frames: {e}")
                break
        
        # Release the video capture object and close any open windows
        cap.release()
        cv2.destroyAllWindows()
    finally:
        print("Video processing complete.")


if __name__ == "__main__":
    main()
