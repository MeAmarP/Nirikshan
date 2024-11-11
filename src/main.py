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


import argparse
parser = argparse.ArgumentParser(description="Get Path to valid Video file..")
parser.add_argument('--fpath',type=str, help="Provide path to Video file" )
parser.add_argument('--record', action='store_true', help="Flag to enable video recording of the output.")
args = parser.parse_args()
path = args.fpath


def main(path_to_vid_file: str, record_video: bool):
    filepath = Path(path_to_vid_file)
    if not filepath.exists():
        # logging.error("Invalid File path. Check if file exists at location.")
        # ! EXIT FROM APPLICATION DUE TO INVALID FILE PATH
        exit(1)
        

    cap =  cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise IOError("Could not open video file.")
    # If successful, continue with processing the video frames
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
        # logging.info("Video recording enabled. Output will be saved as 'output.avi'.")

    
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
                # display_detections(frame=frame, detections=detections)

                # draw tracked objects
                display_tracked_ids(frame=frame, tracked_objects=tracked_objects)

            if len(face_detections) > 0:
                display_faces(frame, face_detections)

            # Record the frame if recording is enabled
            if record_video and out_writer is not None:
                print("+")
                out_writer.write(frame)

            # Display the current frame
            # cv2.imshow('Video', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"An error occurred while trying to process the video frames: {e}")
            traceback.print_exc()
            continue
    
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(path_to_vid_file=path, record_video=args.record)
