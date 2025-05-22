from enum import Enum

# Moved ProcessingMode to the top
class ProcessingMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    DETAILED = "detailed"
    CUSTOM = "custom"

import cv2
import numpy as np
import time
import os
import threading
import concurrent.futures
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path
from functools import lru_cache
import hashlib
import uuid
from PIL import Image
# import torch # Torch might not be needed if heavy GPU features are stubbed
from app.utils.image_utils import ImageUtils
from app.core.config import settings
# Import a STUBBED or working FaceFeatureLevel and FaceFeatureExtractor
from app.services.face_features import FaceFeatureExtractor, FaceFeatureLevel

print("[INFO] Loading a VERY simplified version of preprocessing.py for startup testing. (ProcessingMode moved to top)")

class ImageType(str, Enum):
    PORTRAIT = "portrait"
    GENERAL = "general"
    # Add other types if absolutely necessary for module to load without error

# Minimal ResultCache if needed by stubbed functions
class ResultCache:
    def get(self, path, key): return None
    def set(self, path, key, val): pass

class ImagePreprocessingService:
    def __init__(self):
        self.image_utils = ImageUtils()
        self._cache = ResultCache() # Use stubbed cache
        try:
            # This might still fail if FaceFeatureExtractor init itself has issues even when stubbed
            self._face_feature_extractor = FaceFeatureExtractor()
            print("[STUB_PPS] ImagePreprocessingService: FaceFeatureExtractor STUB INSTANTIATED.")
        except Exception as e:
            self._face_feature_extractor = None
            print(f"[STUB_PPS_WARN] ImagePreprocessingService: Failed to init STUBBED FaceFeatureExtractor: {e}")
        self.max_image_size = settings.MAX_IMAGE_SIZE
        print("[STUB_PPS] ImagePreprocessingService STUB Initialized.")
        
    def process_image(self, image_path: str, mode: ProcessingMode = ProcessingMode.STANDARD, 
                      use_cache: bool = True, face_feature_level: FaceFeatureLevel = FaceFeatureLevel.NORMAL) -> Dict[str, Any]:
        print(f"[STUB_PPS] process_image called for {Path(image_path).name} (stubbed).")
        level_value = face_feature_level.value if isinstance(face_feature_level, Enum) else str(face_feature_level)
        mode_value = mode.value if isinstance(mode, Enum) else str(mode)
        return {
            "original_path": image_path,
            "processing_mode": mode_value,
            "face_feature_level_requested": level_value,
            "created_at": time.time(),
            "error": "ImagePreprocessingService is stubbed.",
            "preprocessed_path": image_path, # Return original path as stub
            "process_time": 0.1
        }

    # Add other methods that tasks.py might try to call from this service, as stubs
    def _read_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        print(f"[STUB_PPS] _read_and_preprocess_image called for {image_path}")
        try: 
            if not os.path.exists(image_path): return None
            return np.zeros((100,100,3), dtype=np.uint8) # Return dummy numpy array
        except: return None

    def _process_face_advanced(self, img_np: np.ndarray, feature_level: FaceFeatureLevel) -> Dict[str, Any]:
        print(f"[STUB_PPS] _process_face_advanced called (stubbed).")
        return {"message": "_process_face_advanced is stubbed"}
    
    def _detect_image_type(self, img: np.ndarray) -> ImageType:
        print(f"[STUB_PPS] _detect_image_type called (stubbed).")
        return ImageType.GENERAL

print("[INFO] VERY simplified version of preprocessing.py LOADED. (ProcessingMode moved to top)") 