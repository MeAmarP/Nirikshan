import json
from pathlib import Path

class AppConfig:
    # Default values can be None or some sensible defaults
    detector_img_size = None
    detector_conf_thresh = None
    detector_nms_thresh = None
    tracker_fps = None
    tracker_track_thresh = None
    tracker_track_buffer = None
    tracker_match_thresh = None

    @classmethod
    def load_config(cls, config_path='default_config.json'):
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Populate class variables
        cls.detector_img_size = config['detector']['img_size']
        cls.detector_conf_thresh = config['detector']['conf_thresh']
        cls.detector_nms_thresh = config['detector']['nms_thresh']
        cls.tracker_fps = config['tracker']['fps']
        cls.tracker_track_thresh = config['tracker']['track_thresh']
        cls.tracker_track_buffer = config['tracker']['track_buffer']
        cls.tracker_match_thresh = config['tracker']['match_thresh']


config_path = str(Path.cwd() / 'src' / 'configs.json')
AppConfig.load_config(config_path)

# Usage after import
# from your_module_name import AppConfig
# print(AppConfig.detector_img_size)
