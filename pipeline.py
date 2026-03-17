import torch
import cv2
from visualizer import Visualizer
from detection_result import DetectionResult
from save_manager import SaveManager

class AnnotatePipeline:
    def __init__(self, source, processor, model, prompt, device, conf, show=False, jump_frames=1, save_manager=None):
        self.source = source
        self.processor = processor
        self.model = model
        self.prompt = prompt
        self.device = device
        self.conf = conf
        self.show = show
        self.jump_frames = jump_frames
        self.save_manager = save_manager
        
    def run(self):
        print("Running pipeline")
        frame_id = 0
        while True:
            frame_id += 1
            frame = self.source.get_frame()
            if frame is None:
                break
            
            if frame_id % self.jump_frames == 0 or frame_id == 1:
                print(frame_id)
                inputs = self.processor(images=frame, text=self.prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=self.conf,
                    text_threshold=0.25,
                    target_sizes=[frame.shape[:2]],
                )

                det_result = DetectionResult(
                    boxes=results[0]["boxes"].cpu().numpy(),
                    scores=results[0]["scores"].cpu().numpy(),
                    labels=results[0]["labels"],
                    text_labels=results[0]["text_labels"]
                )

                if det_result and self.save_manager:
                    self.save_manager.save(frame, det_result, frame_id)

                if self.show:
                    vis_frame = Visualizer.draw_detections(frame.copy(), det_result.boxes, det_result.labels, det_result.scores)
                    key = Visualizer.show(vis_frame)
                    if key == ord('q'):
                        break
                