import cv2
import numpy as np
import traceback
from pathlib import Path
import os

import json

from configs import AppConfig
from detector import ObjectDetector
from tracker.byte_tracker import BYTETracker
from utils import display_detections, display_tracked_ids, display_analytics
from utils import display_faces
from core.analytics import CountAnalytics

def main():

    # Initialize object detector
    yolov3_detector = ObjectDetector(model='yolov3')

    # Init Face Detector
    face_detector = ObjectDetector(model='yunet-face')

    # Initialize object tracker
    tracker = BYTETracker(frame_rate=AppConfig.tracker_fps)  # Assuming 30 fps for now (can be adjusted later)

    # Initialize analytics object
    count_analytics = CountAnalytics()

    try:
    # Open the video file
        cap = cv2.VideoCapture("/data/office.mp4")
    except Exception as e:
        print(f"An error occurred while trying to open the video file: {e}")
        traceback.print_exc()
    else:
        # If successful, continue with processing the video frames
        while True:
            try:
                # Read a frame from the video
                ret, frame = cap.read()
                
                # If the 'ret' value is False (i.e., there are no more frames to read)
                if not ret:
                    print("No frames to read.")
                    break


                # Detect objects
                detections = yolov3_detector.detect(frame)

                # Detect faces
                face_detections = face_detector.detect(frame)

                if len(detections) > 0:

                    # byte-tracker expects detection format as numpy array of shape (N, 5), 
                    # where N is the number of detections and each detection has format [x1, y1, x2, y2, confidence]  
                    np_detections = np.array([np.concatenate((np.array(det['box']).astype(np.float16), np.array([det['confidence']]).astype(np.float16))) for det in detections])
                    
                    tracked_objects = tracker.update(np_detections)
                    
                    # ** Analytics **
                    count_analytics.update(tracked_objects, 'person')
                    display_analytics(frame, count_analytics)

                    # draw detections
                    # display_detections(frame=frame, detections=detections)

                    # draw tracked objects
                    display_tracked_ids(frame=frame, tracked_objects=tracked_objects)

                if len(face_detections) > 0:
                    display_faces(frame, face_detections)

                # Display the current frame
                cv2.imshow('Video', frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"An error occurred while trying to process the video frames: {e}")
                traceback.print_exc()
                break
        
        # Release the video capture object and close any open windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
