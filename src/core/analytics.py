from configs import AppConfig


class CountAnalytics:
    def __init__(self):
        self.name = 'CountAnalytics'
        self.target_class_labels = AppConfig.detector_class_labels
        self.classwise_count_curr = {}  # stores objects to be tracked with their counts
        self.init_counts()  # initialize counts for each class label

    def __str__(self) -> str:
        return " ".join(f"{k} {v}" for k,v in self.classwise_count_curr.items())

    def init_counts(self):        
        for label in self.target_class_labels:
            if label not in self.classwise_count_curr:
                self.classwise_count_curr[label] = 0

    def update(self, tracked_objects, label):            
        # If the object is a target to be tracked...
        if label in self.target_class_labels:
            # Increment its count
            self.classwise_count_curr[label] = len(tracked_objects)

    def get(self):
        return self.classwise_count_curr
