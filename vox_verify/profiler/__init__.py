"""
Vox-Verify Profiler Package
============================
Memory and performance profiling tools for the Vox-Verify pipeline.

Exports:
    MemoryProfiler      - Background-thread RSS/VMS/CPU sampler
    MemorySnapshot      - Timestamped memory reading dataclass
    InferenceProfiler   - Per-pass ONNX inference timer
    InferenceProfile    - Dataclass holding per-stage timings
    FeatureExtractionOptimizer - Vectorised audio preprocessing helpers
    PerformanceAuditor  - End-to-end pipeline auditor
    PerformanceReport   - Structured audit report dataclass
"""

from .memory_audit import (
    MemorySnapshot,
    MemoryProfiler,
    InferenceProfile,
    InferenceProfiler,
    FeatureExtractionOptimizer,
    PerformanceReport,
    PerformanceAuditor,
)

__all__ = [
    "MemorySnapshot",
    "MemoryProfiler",
    "InferenceProfile",
    "InferenceProfiler",
    "FeatureExtractionOptimizer",
    "PerformanceReport",
    "PerformanceAuditor",
]
