import traceback

import cv2
import mediapipe as mp


def check_mediapipe():
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()

        # Initialize OpenCV Video Capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to exit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Face Detection
            results = face_detection.process(rgb_frame)

            # Draw face detections on the frame
            if results.detections:
                for detection in results.detections:
                    # Get the bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('MediaPipe Face Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred while trying to process the video frames: {e}")
        traceback.print_exc()
        break


if __name__ == "__main__":
    check_mediapipe()
