import cv2
import numpy as np

class Color:
    """
    Class for color constants.
    """
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    purple = (128, 0, 128)
    brown = (165, 42, 42)
    gray = (128, 128, 128)
    pink = (255, 192, 203)
    teal = (128, 128, 0)

def display_detections(frame: np.array, detections):
    """Draw raw detector bounding boxes and class labels on the frame."""
    for dets in detections:
        bbox = dets['box']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), Color.green, 2)
        cv2.putText(frame, dets['class_name'], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, Color.orange, 2)

def display_faces(frame: np.array, faces):
    """Draw face bounding boxes and key landmarks."""

    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255),  # left mouth corner
    ]
    for face in faces:
        bbox = face[0:4].astype(np.int32)
        cv2.rectangle(frame, bbox, Color.red, 2, cv2.LINE_AA)

        landmarks = face[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(frame, landmark, 2, landmark_color[idx], 2)


def display_tracked_ids(frame: np.array, tracked_objects):
    """Render tracker-assigned IDs and bounding boxes around tracked objects."""
    if not tracked_objects:
        return

    for obj in tracked_objects:
        bbox = obj.tlwh.astype(np.int32)
        id = obj.track_id
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), Color.cyan, 2)
        cv2.putText(frame, str(id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_DUPLEX, 2, Color.orange, 2)

# function to display the analytics on frame
def display_analytics(frame: np.array, analytics):
    """Render analytics overlays (counts, heatmaps, etc.) onto a frame."""
    if analytics is None:
        return frame

    if isinstance(analytics, (list, tuple)):
        for analytic in analytics:
            frame = display_analytics(frame, analytic)
        return frame

    name = getattr(analytics, 'name', '')

    if name == 'CountAnalytics':
        values = analytics.get()
        for idx, (key, value) in enumerate(sorted(values.items())):
            y_offset = 30 + (idx * 30)
            cv2.putText(
                frame,
                f"{str(key).capitalize()} count: {value}",
                (10, y_offset),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                Color.red,
                2,
            )
        return frame

    if name == 'CrowdDensityAnalytics':

        frame = analytics.render_overlay(
            frame,
            alpha_fg=0.6,
            alpha_overlay=0.4,
            density_threshold=0.5,
            # If you have a Color enum, pass Color.orange / Color.red here:
            # color_low=Color.orange, color_high=Color.red
        )
        return frame


    return frame
