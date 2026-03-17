import os
import cv2
import json
import random
import logging

class SaveManager:

    def __init__(self, dataset_dir="dataset"):
        # Dataset structure
        self.dataset_dir = dataset_dir

        # Dataset split ratios
        self.train_size = 0.7
        self.val_size = 0.2
        self.test_size = 0.1

        # Internal counters
        self.image_id = {"train": 1, "val": 1, "test": 1}
        self.annotation_id = {"train": 1, "val": 1, "test": 1}

        # COCO structures
        self.coco = {
            "train": self._empty_coco(),
            "val": self._empty_coco(),
            "test": self._empty_coco()
        }

        # Define categories once
        self.categories = [
            {
                "id": 1,
                "name": "drone",
                "supercategory": "object"
            }
        ]

        for split in self.coco:
            self.coco[split]["categories"] = self.categories

    def save(self, frame, det_result, frame_id):
        """
        Save image and append annotations to COCO structure.
        """

        split = self._allocate_split()

        filename = f"frame_{frame_id}.jpg"

        images_dir = os.path.join(self.dataset_dir, split, "images")
        os.makedirs(images_dir, exist_ok=True)

        image_path = os.path.join(images_dir, filename)

        # Save image
        cv2.imwrite(image_path, frame)

        height, width = frame.shape[:2]

        image_id = self.image_id[split]

        # Add image entry
        self.coco[split]["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        })

        # Add annotations
        for box in det_result.boxes:

            x0, y0, x1, y1 = map(float, box)

            width_box = x1 - x0
            height_box = y1 - y0

            self.coco[split]["annotations"].append({
                "id": self.annotation_id[split],
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x0, y0, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })

            self.annotation_id[split] += 1

        self.image_id[split] += 1

    def finalize(self):
        """
        Write COCO JSON files to disk.
        """
        logging.info("Finalizing dataset..")
        for split in ["train", "val", "test"]:

            split_dir = os.path.join(self.dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            ann_path = os.path.join(split_dir, "annotations.json")

            with open(ann_path, "w") as f:
                json.dump(self.coco[split], f, indent=2)

    def _allocate_split(self):
        """
        Randomly assign dataset split.
        """

        r = random.random()

        if r < self.train_size:
            return "train"

        elif r < self.train_size + self.val_size:
            return "val"

        return "test"

    def _empty_coco(self):
        """
        Return empty COCO structure.
        """

        return {
            "images": [],
            "annotations": [],
            "categories": []
        }