from configs import AppConfig


class CountAnalytics:
    def __init__(self):
        self.target_class_labels = AppConfig.detector_class_labels
        self.classwise_count = {}  # stores objects to be tracked with their counts
        self.init_counts()  # initialize counts for each class label

    def __str__(self) -> str:
        return " ".join(f"{k} {v}" for k,v in self.classwise_count.items())

    def init_counts(self):        
        for label in self.target_class_labels:
            if label not in self.classwise_count:
                self.classwise_count[label] = 0

    def update(self, detection_output):
        # Iterate over all detected objects
        for obj in detection_output:
            # Get the object's class label
            label = obj['label']
            
            # If the object is a target to be tracked...
            if label in self.target_class_labels:
                # Increment its count
                self.classwise_count[label] += 1
