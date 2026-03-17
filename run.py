import logging

from source_factory import SourceFactory
from config_loader import read_from_config
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import get_device, CleanupManager
from pipeline import AnnotatePipeline
from visualizer import Visualizer

# -----------------------------
# CONSTANTS
# -----------------------------

CONFIG_PATH = "config.json"

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = read_from_config(CONFIG_PATH)

    device = get_device()

    source = SourceFactory.create(source_path=config["source"])
    visualizer = Visualizer()

    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

    pipeline = AnnotatePipeline(
        source=source,
        processor=dino_processor,
        model=dino_model,
        prompt=config["prompt"],
        device=device,
        conf=config["conf"],
        show=config["show"],
        jump_frames=config["jump_frames"]
    )

    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)

    try:
        pipeline.run()
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()
    






if __name__ == "__main__":
    main()