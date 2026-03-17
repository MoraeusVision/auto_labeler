import logging
import torch
import glob
import json
import os

class CleanupManager:
    def __init__(self):
        # Store cleanup functions
        self._tasks = []

    def add(self, func, *args, **kwargs):
        """
        Add a cleanup function to be called later.

        Args:
            func (callable): function to call
            *args, **kwargs: arguments for the function
        """
        self._tasks.append((func, args, kwargs))

    def run(self):
        """Run all registered cleanup tasks."""
        for func, args, kwargs in self._tasks:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.exception("Cleanup failed for %s: %s", func, e)
        self._tasks.clear()
        

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    return device

def reset_dataset_dir():
        """
        Tömmer annotations.json och tar bort alla bilder i images-mapparna för train, val och test.
        """
        
        base_dirs = ["dataset/train", "dataset/val", "dataset/test"]
        for d in base_dirs:
            # Töm annotations.json
            ann_path = os.path.join(d, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump({"images": [], "annotations": [], "categories": []}, f, indent=2)
            # Ta bort alla bilder i images-mappen
            images_dir = os.path.join(d, "images")
            if os.path.exists(images_dir):
                for img_file in glob.glob(os.path.join(images_dir, "*")):
                    try:
                        os.remove(img_file)
                    except Exception:
                        pass