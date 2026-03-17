import os
import cv2
import json

class SaveManager:
    def __init__(self):
        self.train_size = 0.7
        self.val_size = 0.2
        self.test_size = 0.1

    def save(self, frame, det_result, frame_id):
        """
        Spara bild och lägg till annoteringar i COCO-format i annotations.json.
        Filnamn: label + frame_id
        """
        directory = self._allocate_directory()
        if det_result.labels:
            label = det_result.labels[0]
        else:
            label = "unknown"
        filename = f"{label}_{frame_id}.jpg"
        images_dir = os.path.join(directory, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        cv2.imwrite(image_path, frame)

        # Ladda eller skapa annotations.json
        ann_path = os.path.join(directory, "annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                try:
                    coco = json.load(f)
                except Exception:
                    coco = {"images": [], "annotations": [], "categories": []}
        else:
            coco = {"images": [], "annotations": [], "categories": []}

        # Skapa COCO-annotering för denna bild
        height, width = frame.shape[:2]
        image_id = len(coco["images"]) + 1
        coco["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        })
        # Lägg till annotationer
        for i, (box, label) in enumerate(zip(det_result.boxes, det_result.labels)):
            x0, y0, x1, y1 = map(float, box)
            bbox = [x0, y0, x1-x0, y1-y0]
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "score": float(det_result.scores[i]) if det_result.scores is not None else None,
                "area": (x1-x0)*(y1-y0),
                "iscrowd": 0
            })
        # Lägg till kategori om den inte finns
        if not any(cat.get("name") == label for cat in coco["categories"]):
            coco["categories"].append({"id": 1, "name": label})

        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

    def _coco_annotation(self, det_result, filename, shape):
        """
        Skapa COCO-format annotering för en bild.
        """
        height, width = shape[:2]
        coco = {
            "images": [
                {
                    "file_name": filename,
                    "height": height,
                    "width": width,
                    "id": 1
                }
            ],
            "annotations": [],
            "categories": []
        }
        for i, (box, label) in enumerate(zip(det_result.boxes, det_result.labels)):
            x0, y0, x1, y1 = map(float, box)
            bbox = [x0, y0, x1-x0, y1-y0]
            coco["annotations"].append({
                "id": i+1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "score": float(det_result.scores[i]) if det_result.scores is not None else None,
                "area": (x1-x0)*(y1-y0),
                "iscrowd": 0
            })
        # Lägg till kategori
        if det_result.labels:
            coco["categories"].append({"id": 1, "name": det_result.labels[0]})
        return coco

    def _allocate_directory(self):
        """
        Slumpa mapp: train, val eller test.
        """
        import random
        r = random.random()
        if r < self.train_size:
            return "dataset/train"
        elif r < self.train_size + self.val_size:
            return "dataset/val"
        else:
            return "dataset/test"

    def _reset_dataset_dir(self):
        pass