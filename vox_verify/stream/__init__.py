"""
vox_verify.stream
=================
Real-time audio stream processor for near real-time deepfake detection.

Public API
----------
.. code-block:: python

    from vox_verify.stream import (
        AudioBuffer,
        StreamCapture,
        InferenceWorker,
        NoiseEstimator,
        BufferEngine,
        EngineConfig,
        DetectionResult,
    )

Quick start
-----------
.. code-block:: python

    from vox_verify.stream import BufferEngine, EngineConfig, DetectionResult

    config = EngineConfig(
        model_path="path/to/model.onnx",
        sensitivity_threshold=0.5,
    )

    def handle_result(result: DetectionResult) -> None:
        label = "SPOOF" if result.is_spoof else "GENUINE"
        print(f"[{label}] spoof={result.spoof_score:.3f}  latency={result.latency_ms:.1f}ms")

    with BufferEngine(config) as engine:
        engine.on_detection = handle_result
        input("Press Enter to stop …")
"""

from vox_verify.stream.buffer_engine import (
    AudioBuffer,
    BufferEngine,
    DetectionResult,
    EngineConfig,
    InferenceWorker,
    NoiseEstimator,
    StreamCapture,
)

__all__ = [
    "AudioBuffer",
    "BufferEngine",
    "DetectionResult",
    "EngineConfig",
    "InferenceWorker",
    "NoiseEstimator",
    "StreamCapture",
]
