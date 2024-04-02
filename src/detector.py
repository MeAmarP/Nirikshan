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
        self.class_names = self._load_class_names(str(Path.cwd() / AppConfig.path_coco_names))
        self.threshold = AppConfig.detector_conf_thresh
        self.nms_threshold = AppConfig.detector_nms_thresh
        self.input_size = AppConfig.detector_input_size
        self.net = self._load_model()
        self.target_class_id = 0

    def _load_class_names(self, class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def _load_model(self):
        if not (self.model_cfg.exists() and self.model_weights.exists()):
            raise FileNotFoundError('Model files not found')

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        return net

    def detect(self, image, target_class_id=0):
        self.target_class_id = target_class_id
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (224, 224), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

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
                "class_name": self.class_names[class_ids[i]],
                "confidence": confidences[i],
                "box": [x, y, x+w, y+h] #tlbr format
            })

        return detections

if __name__ == "__main__":
    from pathlib import Path
    import os

    model_cfg = str(Path(os.getcwd()) / 'models' / 'yolov3.cfg')
    model_weights = str(Path(os.getcwd()) / 'models' / 'yolov3.weights')
    class_file = str(Path(os.getcwd()) / 'assets' / 'coco.names')
    image_path = str(Path(os.getcwd()) / 'data' / '1.jpg')
    
    detector = ObjectDetector(model_cfg=model_cfg, 
                              model_weights=model_weights,
                              class_file=class_file)
    
    detections = detector.detect(image_path)
    print(detections)

