"""ONNX Runtime session manager and inference service for PixelMend backend."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from src.app.backend.config import Settings, get_settings

logger = logging.getLogger("pixelmend.inference")


class ModelSessionManager:
    """Manages loaded ONNX Runtime inference sessions across all 4 assignment tasks."""

    EXPECTED_MODELS = [
        "task1_universal_ae",
        "task2_classifier",
        "task2_specialist_blur",
        "task2_specialist_occlusion",
        "task2_specialist_salt",
        "task3_soft_moe",
        "task4_generator",
    ]

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.providers: List[str] = []
        self.active_provider: str = "CPUExecutionProvider"

    def _resolve_providers(self) -> List[str]:
        """Detect and configure available ONNX execution providers."""
        available = ort.get_available_providers()
        if self.settings.prefer_cuda and "CUDAExecutionProvider" in available:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.active_provider = "CUDAExecutionProvider"
        else:
            self.providers = ["CPUExecutionProvider"]
            self.active_provider = "CPUExecutionProvider"
        return self.providers

    def _get_model_path_map(self) -> Dict[str, str]:
        """Return mapping of model keys to file paths."""
        base_dir = self.settings.models_dir
        return {
            "task1_universal_ae": os.path.join(base_dir, self.settings.task1_model_name),
            "task2_classifier": os.path.join(base_dir, self.settings.task2_classifier_name),
            "task2_specialist_blur": os.path.join(base_dir, self.settings.task2_blur_name),
            "task2_specialist_occlusion": os.path.join(
                base_dir, self.settings.task2_occlusion_name
            ),
            "task2_specialist_salt": os.path.join(base_dir, self.settings.task2_salt_name),
            "task3_soft_moe": os.path.join(base_dir, self.settings.task3_moe_name),
            "task4_generator": os.path.join(base_dir, self.settings.task4_generator_name),
        }

    def load_all_models(self) -> None:
        """Instantiate all ONNX inference sessions during application lifespan startup."""
        self._resolve_providers()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        path_map = self._get_model_path_map()
        for key, path in path_map.items():
            if os.path.isfile(path):
                try:
                    sess = ort.InferenceSession(path, sess_options=opts, providers=self.providers)
                    self.sessions[key] = sess
                    logger.info(
                        "Loaded ONNX session for %s from %s (%s)",
                        key,
                        path,
                        sess.get_providers()[0],
                    )
                except Exception as exc:
                    logger.error("Failed to load ONNX model %s from %s: %s", key, path, exc)
            else:
                logger.warning("ONNX model file not found for %s at %s", key, path)

    def get_session(self, model_key: str) -> ort.InferenceSession:
        """Retrieve loaded InferenceSession or raise KeyError."""
        if model_key not in self.sessions:
            raise KeyError(
                f"Model '{model_key}' is not loaded. Available models: {list(self.sessions.keys())}"
            )
        return self.sessions[model_key]

    def has_model(self, model_key: str) -> bool:
        """Check if a specific model session is loaded and ready."""
        return model_key in self.sessions

    def run_inference(
        self,
        model_key: str,
        input_feed: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """Execute inference on the specified loaded ONNX session."""
        sess = self.get_session(model_key)
        return sess.run(None, input_feed)

    def get_health_status(self) -> Tuple[str, str, Dict[str, bool]]:
        """Return overall health status ('ok'/'degraded'), active provider, and loaded model map."""
        model_status = {key: (key in self.sessions) for key in self.EXPECTED_MODELS}
        all_loaded = all(model_status.values())
        status = "ok" if all_loaded else "degraded"
        return status, self.active_provider, model_status

    def clear(self) -> None:
        """Release all model sessions from memory."""
        self.sessions.clear()


# Global singleton instance for lifespan and dependency injection
session_manager = ModelSessionManager()
