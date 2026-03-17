import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def draw_detections(frame, boxes, labels, scores=None, color=(0,255,0)):
        """
        Draw bounding boxes and labels on the frame.
        Args:
            frame: np.ndarray (BGR)
            boxes: np.ndarray (N, 4) in XYXY format
            labels: list of str
            scores: list or np.ndarray (optional)
            color: tuple (B, G, R)
        """
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = labels[i] if labels else ""
            score = f" {scores[i]:.2f}" if scores is not None else ""
            text = f"{label}{score}"
            if text.strip():
                cv2.putText(frame, text, (x0, max(y0-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    @staticmethod
    def show(frame, window_name="Detection", delay=1):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        return key

    @staticmethod
    def cleanup(window_name="Detection"):
        cv2.destroyWindow(window_name)
