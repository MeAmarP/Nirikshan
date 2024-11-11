# Nirikshan (Supervision)
## Real Time Video Analytics Pipeline using object detection and tracking 

    *NOTE --> ByteTracker uses python-package lap, which wont work for python-version > 3.10*

Nirikshan aims to provide video analytics on video sources like RTSP stream or video file using deep learning models.

- Video Source
  - [x] Video file
  - [ ] RTSP Stream
- DNN Models
  - [x] object detection: yolo-V3/V4
  - [ ] Pose Estimation: MediaPipe
  - [x] Face Detection
- [x] Object Tracker: ByteTracker
- [x] OpenCV for Video Processing and DNN for Inference
- [ ] NVIDIA Triton for Inference
- [ ] User Interface
- [x] Python

### Analytics
- Class: **Person**
  - [x] Count 
  - ~~Emotion~~ NotImplemented
  - ~~Age Category~~ NotImplemented
    - ~~Young Adults~~NotImplemented
    - ~~Middle Aged~~NotImplemented
    - ~~Older Adults~~NotImplemented
  - Action
    - Smoking
    - Fighting
- Class: **Vehicle**
   - Count
   - Type (Car, Bus, Bike)
   - Color
   - Brand
   - LPR
- Class: **Animal**
  - Count
  - Species

### How to run
1. Install dependencies using conda, use conda env file `myenv.yml`
3. Download models from below links
    - [YOLO Models](https://github.com/AlexeyAB/darknet?tab=readme-ov-file#pre-trained-models)
    - [Yunet-face](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
4. Make sure conda env is activated.
5. To Run application, Use command `make run FPATH=data/palace.mp4`

### FUTURE SCOPE (Items in the list are in consideration, not finalized though)
- Action recognition in videos.
- Support for multiple video sources (IP Cameras, Local Files)
- Dockerize Analytics
- GPU Support for faster inference
- User Interface for visualizing analytics results

### References/citations
- yolov3: Radmon et al. "YOLOv3: An Incremental Improvement"
- yolov4: Bochkovskiy et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection"
- ByteTrack: Zhang,Yifu et al. Multi-Object Tracking by Associating Every Detection Box.
- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- PaddleDetection: https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/README_en.md
