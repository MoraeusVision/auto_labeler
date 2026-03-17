import logging

from source_factory import SourceFactory
from config_loader import read_from_config
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import get_device, CleanupManager, reset_dataset_dir
from pipeline import AnnotatePipeline
from visualizer import Visualizer
from save_manager import SaveManager

# -----------------------------
# CONSTANTS
# -----------------------------

CONFIG_PATH = "config.json"

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = read_from_config(CONFIG_PATH)
    if config["reset_dataset_dir"]:
        reset_dataset_dir()

    device = get_device()

    source = SourceFactory.create(source_path=config["source"])
    visualizer = Visualizer()
    save_manager = SaveManager()

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
        jump_frames=config["jump_frames"],
        save_manager=save_manager
    )

    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)
    if config["save"]:
        cleanup.add(save_manager.finalize)

    try:
        pipeline.run()
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()
    






if __name__ == "__main__":
    main()