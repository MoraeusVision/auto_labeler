from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class DetectionResult:
    boxes: np.ndarray      # shape (N, 4)
    scores: np.ndarray     # shape (N,)
    labels: List[str]      # length N
    text_labels: List[str] # length N