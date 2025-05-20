from enum import Enum
import numpy as np

print("[INFO] Loading a VERY simplified version of face_features.py for startup testing.")

class FaceFeatureLevel(Enum):
    MINIMAL = 1
    NORMAL = 2
    DETAIL = 3

class FaceFeatureExtractor:
    def __init__(self, level: FaceFeatureLevel = FaceFeatureLevel.NORMAL):
        self.level = level
        print(f"[STUB] FaceFeatureExtractor initialized with level {self.level.name} (stubbed, no mediapipe).")

    def extract_features(self, image_path: str, level: FaceFeatureLevel = None) -> dict:
        print(f"[STUB] extract_features called for {image_path} (stubbed).")
        return {
            "landmarks": [],
            "blendshapes": [],
            "transformation_matrix": [],
            "error": "FaceFeatureExtractor is stubbed for testing.",
            "face_feature_level_used": (level or self.level).name
        }

    def get_face_mesh_points(self, image_path: str) -> list:
        print(f"[STUB] get_face_mesh_points called for {image_path} (stubbed).")
        return []

    def get_iris_landmarks(self, image_path: str) -> list:
        print(f"[STUB] get_iris_landmarks called for {image_path} (stubbed).")
        return [] 