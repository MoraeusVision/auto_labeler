from detection_result import DetectionResult

def box_aspect_ratio_filter(det_results, max_ratio_deviation=0.5):
    boxes = det_results.boxes
    keep_indices = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if h == 0:
            continue

        aspect_ratio = w / h

        if (1 - max_ratio_deviation) <= aspect_ratio <= (1 + max_ratio_deviation):
            keep_indices.append(i)

    return DetectionResult(
        boxes=det_results.boxes[keep_indices],
        scores=det_results.scores[keep_indices],
        labels=[det_results.labels[i] for i in keep_indices],
        text_labels=[det_results.text_labels[i] for i in keep_indices]
    )

def box_size_filter(det_results, frame_shape, min_area_ratio=0.001, max_area_ratio=0.5):
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_w * frame_h

    keep_indices = []

    for i, box in enumerate(det_results.boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        box_area = w * h

        area_ratio = box_area / frame_area

        if min_area_ratio <= area_ratio <= max_area_ratio:
            keep_indices.append(i)

    return DetectionResult(
        boxes=det_results.boxes[keep_indices],
        scores=det_results.scores[keep_indices],
        labels=[det_results.labels[i] for i in keep_indices],
        text_labels=[det_results.text_labels[i] for i in keep_indices]
    )