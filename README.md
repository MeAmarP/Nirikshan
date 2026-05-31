# Nirikshan (Supervision)
## Real Time Video Analytics Pipeline using object detection and tracking 

    *NOTE --> ByteTracker uses python-package lap, which wont work for python-version > 3.10*
    *NOTE --> Using PaddlePaddle CPU Version 2.6.2*

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
  - [x] Crowd density heatmap
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

**Local environment**
1. Install dependencies using conda with the provided environment file `myenv.yml`.
2. Download the required model weights into the paths referenced in `src/configs.json`:
    - [YOLO Models](https://github.com/AlexeyAB/darknet?tab=readme-ov-file#pre-trained-models)
    - [Yunet-face](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
3. Activate the `myenv` conda environment.
4. Run the application with `make run FPATH=data/palace.mp4` (replace the path with your video source).

**Docker**
- Build the image: `docker build -t nirikshan:latest .`
- Run analytics (the container entrypoint is `python src/main.py`, so pass CLI args directly):
  `docker run -it --gpus all -v $(pwd)/data:/workspace/data nirikshan:latest --fpath /workspace/data/sample.mp4`
- Mount or bake in the model weight directories so the paths in `src/configs.json` remain valid inside the container.

### Crowd density heatmaps
- Each frame is clustered into crowd "zones" using OpenCV k-means, and the resulting heatmap is blended onto the video feed in real time.
- Tweak the behaviour (zone count, heatmap resolution, decay) via the `analytics.crowd_density` section in `src/configs.json`.
- The overlay appears automatically when people are detected; the count overlay still shows per-class tallies alongside the heatmap.

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
