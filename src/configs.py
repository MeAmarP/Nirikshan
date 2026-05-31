import json
from pathlib import Path

class AppConfig:
    # Default values can be None or some sensible defaults
    yolov3_weights = None
    yolov3_cfg = None
    yolov4_weights = None
    yolov4_cfg = None
    detector_img_size = None
    detector_conf_thresh = None
    detector_nms_thresh = None
    tracker_fps = None
    tracker_track_thresh = None
    tracker_track_buffer = None
    tracker_match_thresh = None
    coco_names = None
    crowd_density_zones = 4
    crowd_density_decay = 0.85
    crowd_density_heatmap_resolution = (64, 64)
    crowd_density_min_detections = 1

    @classmethod
    def load_config(cls, config_path='default_config.json'):
        with open(config_path, 'r') as file:
            config = json.load(file)
        # others
        cls.path_coco_names = str(Path(config['others']['coco_names']))
        # yolov3
        cls.yolov3_weights = str(Path(config['yolov3_model']['weights']))
        cls.yolov3_cfg = str(Path(config['yolov3_model']['config']))
        # yolov4
        cls.yolov4_weights = str(Path(config['yolov4_model']['weights']))
        cls.yolov4_cfg = str(Path(config['yolov4_model']['config']))
        # yolo-detector
        cls.detector_class_labels = config['detector']['class_labels']
        cls.detector_input_size = config['detector']['input_size']
        cls.detector_conf_thresh = config['detector']['conf_thresh']
        cls.detector_nms_thresh = config['detector']['nms_thresh']
        # yunet-face 
        cls.yunet_weights = str(Path(config['yunet-face']['weights']))
        cls.yunet_input_size = config['yunet-face']['input_size']  # (h, w) tuple
        cls.yunet_conf_thresh = config['yunet-face']['conf_thresh']
        cls.yunet_nms_thresh = config['yunet-face']['nms_thresh']
        # tracker
        cls.tracker_fps = config['tracker']['fps']
        cls.tracker_track_thresh = config['tracker']['track_thresh']
        cls.tracker_track_buffer = config['tracker']['track_buffer']
        cls.tracker_match_thresh = config['tracker']['match_thresh']
        # analytics
        analytics_cfg = config.get('analytics', {})
        crowd_cfg = analytics_cfg.get('crowd_density', {})
        cls.crowd_density_zones = crowd_cfg.get('zones', cls.crowd_density_zones)
        cls.crowd_density_decay = crowd_cfg.get('decay', cls.crowd_density_decay)
        cls.crowd_density_min_detections = crowd_cfg.get('min_detections', cls.crowd_density_min_detections)
        resolution = crowd_cfg.get('heatmap_resolution', list(cls.crowd_density_heatmap_resolution))
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            cls.crowd_density_heatmap_resolution = (int(resolution[0]), int(resolution[1]))


config_path = str(Path.cwd() / 'src' / 'configs.json')
AppConfig.load_config(config_path)
