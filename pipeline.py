import torch
import cv2
from visualizer import Visualizer

class AnnotatePipeline:
    def __init__(self, source, processor, model, prompt, device):
        self.source = source
        self.processor = processor
        self.model = model
        self.prompt = prompt
        self.device = device
        
    def run(self):
        print("Running pipeline")
        while True:
            frame = self.source.get_frame()
            if frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=frame_rgb, text=self.prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            print(f"Looking for: {self.prompt}")

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.3,
                text_threshold=0.25,
                target_sizes=[frame_rgb.shape[:2]],
            )

            # Visualization
            boxes = results[0]["boxes"].cpu().numpy() if results[0]["boxes"].numel() > 0 else []
            scores = results[0]["scores"].cpu().numpy() if results[0]["scores"].numel() > 0 else []
            labels = results[0]["labels"] if results[0]["labels"] else []

            vis_frame = Visualizer.draw_detections(frame.copy(), boxes, labels, scores)
            key = Visualizer.show(vis_frame)
            if key == ord('q'):
                break
        Visualizer.close()