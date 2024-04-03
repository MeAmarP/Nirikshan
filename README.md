# Nirikshan (Supervision)
## Real Time Video Analytics Pipeline using object detection and tracking 

Nirikshan aims to provide video analytics on video sources like RTSP stream or video file.

- DNN Models
  - object detection: yolo-V3/V4
  - Pose Estimation: MediaPipe
  - Face Detection
- Object Tracker: ByteTracker
- OpenCV for Video Processing and DNN for Inference
- User Interface
- Python

### Analytics
- Class: **Person**
  - [x] Count 
  - Emotion
  - Age Category
    - Young Adults
    - Middle Aged
    - Older Adults
  - Action
    - Smoking
    - Fighting
    - Patient monitoring for fall
- Class: **Vehicle**
   - Count
   - Type (Car, Bus, Bike)
   - Color
   - Brand
   - LPR
- Class: **Animal**
  - Count
  - Species
  
### FUTURE SCOPE (Items in the list are in consideration, not finalized though)
- Action recognition in videos.
- Support for multiple video sources (IP Cameras, Local Files)
- Dockerize Analytics
- GPU Support for faster inference
- User Interface for visualizing analytics results

### References
- yolov3:
- yolov4: 
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box. Zhang, Yifu, et al. Proceedings of the European Conference on Computer Vision (ECCV). 2022.
