"""
vox_verify.engine
=================
Quantization and conversion engine for the VoxVerify deepfake detection system.

Provides utilities to export PyTorch models to ONNX format and apply
INT8/FP16 quantization for fast inference on consumer hardware.
"""

from .quantize import (
    export_to_onnx,
    quantize_fp16,
    quantize_int8_dynamic,
    quantize_int8_static,
    benchmark_inference,
    AudioCalibrationDataReader,
)

__all__ = [
    "export_to_onnx",
    "quantize_fp16",
    "quantize_int8_dynamic",
    "quantize_int8_static",
    "benchmark_inference",
    "AudioCalibrationDataReader",
]
