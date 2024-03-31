import cv2
import numpy as np

# TODO - Detector using DarkNet and CUDA support
# TODO - Detector using NVIDIA Triton Server

class ObjectDetector:
    def __init__(self, model_cfg='yolov3.cfg', model_weights='yolov3.weights', class_file='coco.names', threshold=0.2, nms_threshold=0.3):
        self.model_cfg = model_cfg
        self.model_weights = model_weights
        self.class_names = self._load_class_names(class_file)
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.net = self._load_model()
        self.target_class_id = 0

    def _load_class_names(self, class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def _load_model(self):
        net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_OPENCL for GPU
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

