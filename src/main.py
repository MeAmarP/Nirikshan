import traceback
from pathlib import Path

import cv2
import numpy as np

from configs import AppConfig
from core.analytics import CountAnalytics
from detector import ObjectDetector
from tracker.byte_tracker import BYTETracker
from utils import (display_analytics, display_detections, display_faces,
                   display_tracked_ids)


def main(path_to_vid_file: str):
    filepath = Path(path_to_vid_file)
    if not filepath.exists():
        print("Invalid File path, Check if file exists at location???")
        # ! EXIT FROM APPLICATION DUE TO INVALID FILE PATH
        exit(1)
        

    with cv2.VideoCapture(str(filepath)) as cap:
        if not cap.isOpened():
            raise IOError("Could not open video file.")
        # If successful, continue with processing the video frames
        
        # ------------------ init components -------------------- 
        # Initialize object (Person) detector
        yolov3_detector = ObjectDetector(model='yolov3')

        # Init Face Detector
        face_detector = ObjectDetector(model='yunet-face')

        # Initialize object tracker
        tracker = BYTETracker(frame_rate=AppConfig.tracker_fps)  # Assuming 30 fps for now (can be adjusted later)

        # Initialize analytics object
        count_analytics = CountAnalytics()
        # -------------------------------------------------------

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
                    display_detections(frame=frame, detections=detections)

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
                continue
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get Path to valid Video file..")
    parser.add_argument('fpath',type=str, help="Provide path to Video file" )
    args = parser.parse_args()
    fpath = args.path
    main(path_to_vid_file=fpath)
