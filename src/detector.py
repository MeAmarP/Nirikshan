from pathlib import Path
import cv2
import numpy as np

from configs import AppConfig
# TODO - Detector using DarkNet and CUDA support
# TODO - Detector using NVIDIA Triton Server

class ObjectDetector:
    def __init__(self, model: str='yolov3',):
        if model == 'yolov3':
            self.model_cfg = Path.cwd() / AppConfig.yolov3_cfg
            self.model_weights = Path.cwd() / AppConfig.yolov3_weights
        if model == 'yolov4':
            self.model_cfg = Path.cwd() / AppConfig.yolov4_cfg
            self.model_weights = Path.cwd() / AppConfig.yolov4_weights
        if model == 'yunet-face':
            self.model_cfg = Path.cwd() / AppConfig.yunet_weights
        self.class_names = self._load_class_names(str(Path.cwd() / AppConfig.path_coco_names))
        self.threshold = AppConfig.detector_conf_thresh
        self.nms_threshold = AppConfig.detector_nms_thresh
        self.input_size = AppConfig.detector_input_size
        self.net = self._load_model()
        self.target_class_id = 0

        if self.model_name == 'yunet-face':
            self.model_cfg = Path.cwd() / AppConfig.yunet_weights
            self.yunet_input_size = AppConfig.yunet_input_size
            self.yunet_conf_thresh = AppConfig.yunet_conf_thresh
            self.yunet_nms_thresh = AppConfig.yunet_nms_thresh
            self.ynet = self._load_model()
        

    def _load_class_names(self, class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
            class_id_names = {label:id for id, label in enumerate(class_names)}
        return class_id_names

    def _load_model(self):
        if self.model_name == 'yunet-face':
            net = cv2.FaceDetectorYN.create(str(self.model_cfg),
                                            "",
                                            (640,640),
                                            self.yunet_conf_thresh,
                                            self.yunet_nms_thresh,
                                            backend_id=0,
                                            target_id=0)
        return net

    def detect(self, image, target_class_labels: list):
        self.target_class_id = [self.class_names[label] for label in target_class_labels]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        # ! TODO Handle case where no objects detected by the model, return empty list or something
        return self._process_outputs(outputs, image)

    def _process_outputs(self, outputs, image):
        height, width = image.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.threshold and class_id == self.target_class_id:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.nms_threshold)

        detections = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            detections.append({
                "class_id": class_ids[i],
                "class_name": [k for k,v in self.class_names.items() if v == self.target_class_id],
                "confidence": confidences[i],
                "box": [x, y, x+w, y+h] #tlbr format
            })

        return detections

    def detect(self, image):
        if self.model_name == 'yolov3':
            return self._yolo_detect(image)
        if self.model_name == 'yolov4':
            return self._yolo_detect(image)
        if self.model_name == 'yunet-face':
            return self._face_yunet_detect(image)
if __name__ == "__main__":
    # TODO Update this to get paths from App config file
    import os
    image_path = str(Path(os.getcwd()) / 'data' / '1.jpg')
    
    detector = ObjectDetector()
    img = cv2.imread(image_path)
    detects = detector.detect(img, AppConfig.detector_class_labels)
    print(detects)

