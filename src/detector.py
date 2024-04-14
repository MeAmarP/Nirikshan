from pathlib import Path
import cv2
import numpy as np

from configs import AppConfig
# TODO - Detector using DarkNet and CUDA support
# TODO - Detector using NVIDIA Triton Server

class ObjectDetector:
    def __init__(self, model: str='yolov3',):
        self.class_names = None
        self.target_class_labels = None
        self.model_cfg = None
        self.model_weights = None
        self.model_name = model
        # ! TODO Modify config.json such a that yolov3/4 have their own parameters 
        # ! Class labels, input size, conf_thresh, nms_thresh etc.
        # ? TODO find better way to manage repeated code
        if self.model_name == 'yolov3':
            self.model_cfg = str(Path.cwd() / AppConfig.yolov3_cfg)
            
            self.model_weights = str(Path.cwd() / AppConfig.yolov3_weights)
            
            # load coco names and get all supported class_labels by model
            self.class_names = self._load_class_names(str(Path.cwd() / AppConfig.path_coco_names))
            
            # load target class labels that need to detected from config
            self.target_class_labels=AppConfig.detector_class_labels
            
            # Get target IDs from class labels we loaded , this will be list of IDs
            self.target_class_id = [self.class_names[label] for label in self.target_class_labels]

            # load detection threshold from config
            self.threshold = AppConfig.detector_conf_thresh
            # load nms threshold from config
            self.nms_threshold = AppConfig.detector_nms_thresh
            # load input size from config
            self.input_size = AppConfig.detector_input_size
            # load model
            self.net = self._load_model()
            self.target_class_id = 0

        if self.model_name == 'yolov4':
            self.model_cfg = str(Path.cwd() / AppConfig.yolov4_cfg)
            self.model_weights = str(Path.cwd() / AppConfig.yolov4_weights)
            # load coco names and get all supported class_labels by model
            self.class_names = self._load_class_names(str(Path.cwd() / AppConfig.path_coco_names))
            
            # load target class labels that need to detected from config
            self.target_class_labels=AppConfig.detector_class_labels
            
            # Get target IDs from class labels we loaded , this will be list of IDs
            self.target_class_id = [self.class_names[label] for label in self.target_class_labels]
            
            # load detection threshold from config
            self.threshold = AppConfig.detector_conf_thresh
            # load nms threshold from config
            self.nms_threshold = AppConfig.detector_nms_thresh
            # load input size from config
            self.input_size = AppConfig.detector_input_size
            # load model
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
        # if not Path(self.model_cfg).exists():
        #     raise FileNotFoundError('Model cfg files not found')
        # if not Path(self.model_weights).exists():
        #     raise FileNotFoundError('Model weights not found')
        if 'yolo' in self.model_name:
            net = cv2.dnn.readNet(str(self.model_weights), str(self.model_cfg))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            return net
        
        if self.model_name == 'yunet-face':
            net = cv2.FaceDetectorYN.create(model = str(self.model_cfg),
                                            config = "",
                                            input_size = (320,320),
                                            score_threshold=self.yunet_conf_thresh,
                                            nms_threshold=self.yunet_nms_thresh,
                                            top_k=5000,
                                            backend_id=0,
                                            target_id=0)
            net.setInputSize((320, 320))
            return net
    
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

    def _yolo_detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        # ! TODO Handle case where no objects detected by the model, return empty list or something
        return self._process_outputs(outputs, image)
    
    def _face_yunet_detect(self, image):
        self.ynet.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.ynet.detect(image)
        faces = faces if faces is not None else []
        return faces
    
    def detect(self, image):
        if self.model_name == 'yolov3':
            return self._yolo_detect(image)
        if self.model_name == 'yolov4':
            return self._yolo_detect(image)
        if self.model_name == 'yunet-face':
            return self._face_yunet_detect(image)

if __name__ == "__main__":
    print("Starting object detection")

