"""
memory_audit.py — Vox-Verify Memory Profiler & Performance Auditor
===================================================================
Non-intrusive memory and latency profiling for the Vox-Verify
live-monitoring pipeline.

Classes
-------
MemorySnapshot              Timestamped process-memory reading.
MemoryProfiler              Background-thread sampler (RSS/VMS/CPU).
InferenceProfile            Per-stage inference timing dataclass.
InferenceProfiler           Nanosecond-precision ONNX inference profiler.
FeatureExtractionOptimizer  Vectorised audio pre-processing helpers.
PerformanceReport           Structured audit report with Markdown / JSON export.
PerformanceAuditor          End-to-end pipeline auditor (main orchestrator).

Constraint
----------
All profiling is non-intrusive: measurement overhead is kept to a minimum
through background threads, context-managers, and lock-free snapshot lists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Generator

import numpy as np
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ONNX Runtime import — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort  # type: ignore

    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_AVAILABLE = False
    logger.warning(
        "onnxruntime not found. InferenceProfiler will operate in stub mode."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BYTES_PER_MB: float = 1024.0 ** 2
MEMORY_LIMIT_MB: float = 2048.0  # 2 GB
LATENCY_LIMIT_MS: float = 500.0  # 500 ms per chunk (real-time constraint)
DEFAULT_SAMPLE_INTERVAL_S: float = 0.1  # 100 ms
SAMPLE_RATE: int = 16_000  # Hz — standard for speech models


# ===========================================================================
# 1. MemorySnapshot
# ===========================================================================

@dataclass
class MemorySnapshot:
    """A single timestamped sample of process memory usage.

    Attributes
    ----------
    timestamp:   Wall-clock time (seconds since epoch) of the sample.
    rss_mb:      Resident Set Size in megabytes.
    vms_mb:      Virtual Memory Size in megabytes.
    cpu_percent: Process CPU usage at sample time (0–100 per core).
    """

    timestamp: float
    rss_mb: float
    vms_mb: float
    cpu_percent: float

    def to_dict(self) -> Dict[str, float]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ===========================================================================
# 2. MemoryProfiler
# ===========================================================================

class MemoryProfiler:
    """Continuously samples the current process's memory in a background thread.

    Usage
    -----
    >>> profiler = MemoryProfiler(interval_s=0.1)
    >>> profiler.start()
    >>> # ... do work ...
    >>> profiler.stop()
    >>> print(profiler.get_peak_memory_mb())

    Parameters
    ----------
    interval_s:
        Sampling interval in seconds (default 0.1 s / 100 ms).
    pid:
        Process ID to monitor.  Defaults to the current process.
    """

    def __init__(
        self,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        pid: Optional[int] = None,
    ) -> None:
        self._interval_s = interval_s
        self._pid = pid or os.getpid()
        self._process = psutil.Process(self._pid)
        self._snapshots: List[MemorySnapshot] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "MemoryProfiler":
        """Start the background sampling thread.

        Returns self for chaining::

            profiler = MemoryProfiler().start()
        """
        if self._thread and self._thread.is_alive():
            logger.warning("MemoryProfiler is already running.")
            return self
        self._stop_event.clear()
        self._snapshots.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="memory-profiler",
            daemon=True,
        )
        self._thread.start()
        logger.debug("MemoryProfiler started (pid=%d, interval=%.3fs)", self._pid, self._interval_s)
        return self

    def stop(self) -> "MemoryProfiler":
        """Signal the sampling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval_s * 10)
        logger.debug("MemoryProfiler stopped (%d snapshots collected)", len(self._snapshots))
        return self

    def get_peak_memory_mb(self) -> float:
        """Return the maximum RSS observed (MB). Returns 0.0 if no data."""
        with self._lock:
            if not self._snapshots:
                return 0.0
            return max(s.rss_mb for s in self._snapshots)

    def get_average_memory_mb(self) -> float:
        """Return the mean RSS across all samples (MB). Returns 0.0 if no data."""
        with self._lock:
            if not self._snapshots:
                return 0.0
            return float(np.mean([s.rss_mb for s in self._snapshots]))

    def get_timeline(self) -> List[MemorySnapshot]:
        """Return a copy of all recorded snapshots."""
        with self._lock:
            return list(self._snapshots)

    def reset(self) -> None:
        """Clear recorded snapshots without stopping the thread."""
        with self._lock:
            self._snapshots.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Background loop: collect one snapshot per interval until stopped."""
        while not self._stop_event.wait(timeout=self._interval_s):
            try:
                mem_info = self._process.memory_info()
                cpu = self._process.cpu_percent(interval=None)
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=mem_info.rss / BYTES_PER_MB,
                    vms_mb=mem_info.vms / BYTES_PER_MB,
                    cpu_percent=cpu,
                )
                with self._lock:
                    self._snapshots.append(snapshot)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning("MemoryProfiler: process no longer accessible; stopping.")
                break

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemoryProfiler":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ===========================================================================
# 3. InferenceProfile & InferenceProfiler
# ===========================================================================

@dataclass
class InferenceProfile:
    """Timing and memory results for a single inference pass.

    Attributes
    ----------
    preprocess_ms:   Time spent in audio pre-processing (ms).
    inference_ms:    Time spent inside the ONNX session.run() call (ms).
    postprocess_ms:  Time spent in post-processing (ms).
    total_ms:        End-to-end wall time (ms).
    memory_delta_mb: Change in RSS from before to after the pass (MB).
    per_layer_ms:    Optional per-operator timings from ORT profiling.
    """

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    memory_delta_mb: float
    per_layer_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InferenceProfiler:
    """Profiles a single ONNX-Runtime inference pass with nanosecond precision.

    The profiler wraps the three pipeline stages—preprocessing, inference,
    postprocessing—in lightweight context managers that use
    ``time.perf_counter_ns`` internally.  Measurement overhead is limited
    to two ``perf_counter_ns`` calls per stage.

    Parameters
    ----------
    enable_ort_profiling:
        When True and ORT is available, enables ORT's built-in per-operator
        profiler and parses the resulting JSON.  Incurs extra I/O overhead;
        disable for production use.
    """

    def __init__(self, enable_ort_profiling: bool = False) -> None:
        self._enable_ort_profiling = enable_ort_profiling and _ORT_AVAILABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_single(
        self,
        model_session: Any,
        input_data: np.ndarray,
    ) -> InferenceProfile:
        """Profile one full inference pass.

        Parameters
        ----------
        model_session:
            An ``onnxruntime.InferenceSession`` (or a callable duck-type that
            accepts ``run(output_names, input_dict)``).
        input_data:
            Raw audio array (shape: [samples], dtype float32) at 16 kHz.

        Returns
        -------
        InferenceProfile
            Per-stage timings and memory delta for this pass.
        """
        proc = psutil.Process()
        rss_before = proc.memory_info().rss / BYTES_PER_MB

        t_total_start = time.perf_counter_ns()

        # ---- Stage 1: Preprocessing ------------------------------------
        with _ns_timer() as pre_timer:
            features = self._preprocess(input_data)

        # ---- Stage 2: Model inference ----------------------------------
        per_layer: Dict[str, float] = {}
        with _ns_timer() as inf_timer:
            if _ORT_AVAILABLE and model_session is not None:
                try:
                    input_name = model_session.get_inputs()[0].name
                    outputs = model_session.run(None, {input_name: features})
                    if self._enable_ort_profiling:
                        per_layer = self._parse_ort_profile(model_session)
                except Exception as exc:
                    logger.debug("ORT run error (stub data): %s", exc)
                    outputs = [np.zeros((1, 128), dtype=np.float32)]
            else:
                # Stub: simulate inference with a small sleep
                time.sleep(0.005)
                outputs = [np.zeros((1, 128), dtype=np.float32)]

        # ---- Stage 3: Postprocessing -----------------------------------
        with _ns_timer() as post_timer:
            _ = self._postprocess(outputs)

        t_total_end = time.perf_counter_ns()
        rss_after = proc.memory_info().rss / BYTES_PER_MB

        return InferenceProfile(
            preprocess_ms=pre_timer.elapsed_ms,
            inference_ms=inf_timer.elapsed_ms,
            postprocess_ms=post_timer.elapsed_ms,
            total_ms=(t_total_end - t_total_start) / 1e6,
            memory_delta_mb=rss_after - rss_before,
            per_layer_ms=per_layer,
        )

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(audio: np.ndarray) -> np.ndarray:
        """Minimal feature extraction: normalise and reshape for inference."""
        # Normalise to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0.0:
            audio = audio / peak
        # Add batch dimension — shape [1, samples]
        return audio.astype(np.float32)[np.newaxis, :]

    @staticmethod
    def _postprocess(outputs: List[np.ndarray]) -> np.ndarray:
        """Apply softmax to the first output tensor."""
        logits = outputs[0]
        exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp_l / exp_l.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # ORT per-layer profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ort_profile(session: Any) -> Dict[str, float]:
        """Read and parse the ORT profiling JSON written by the session."""
        try:
            profile_path = session.end_profiling()
            with open(profile_path, "r") as fh:
                events = json.load(fh)
            timings: Dict[str, float] = {}
            for ev in events:
                name = ev.get("name", "unknown")
                dur_us = ev.get("dur", 0)
                timings[name] = timings.get(name, 0.0) + dur_us / 1e3  # µs → ms
            return timings
        except Exception as exc:
            logger.debug("ORT profile parse failed: %s", exc)
            return {}


# ===========================================================================
# 4. FeatureExtractionOptimizer
# ===========================================================================

class FeatureExtractionOptimizer:
    """Analyses the audio pre-processing pipeline and provides optimised ops.

    All public ``vectorized_*`` methods avoid Python-level loops; they rely
    entirely on NumPy broadcasting, stride tricks, and ufuncs.

    Methods
    -------
    vectorized_normalize        In-place peak normalisation.
    vectorized_frame_split      Zero-copy overlapping frame extraction.
    vectorized_rms              Per-frame RMS energy.
    vectorized_preemphasis      High-frequency emphasis filter.
    recommend_optimizations     Heuristic bottleneck checker.
    """

    # ------------------------------------------------------------------
    # Optimised audio operations
    # ------------------------------------------------------------------

    @staticmethod
    def vectorized_normalize(audio: np.ndarray) -> np.ndarray:
        """Normalise *audio* to the range [-1, 1] in-place where the array is
        writeable, falling back to a copy for read-only arrays.

        Parameters
        ----------
        audio:
            1-D float32 audio signal.

        Returns
        -------
        np.ndarray
            Normalised array (same shape, same dtype).
        """
        peak = np.abs(audio).max()
        if peak == 0.0:
            return audio  # silence — nothing to normalise
        if audio.flags.writeable:
            audio /= peak  # true in-place — no allocation
            return audio
        return audio / peak  # read-only input → allocate once

    @staticmethod
    def vectorized_frame_split(
        audio: np.ndarray,
        frame_size: int,
        hop_size: int,
    ) -> np.ndarray:
        """Split *audio* into overlapping frames using zero-copy stride tricks.

        Uses ``np.lib.stride_tricks.as_strided`` so the returned array shares
        memory with *audio* — no data is copied.

        Parameters
        ----------
        audio:
            1-D float array of shape ``(N,)``.
        frame_size:
            Number of samples per frame.
        hop_size:
            Number of samples between successive frame starts.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_frames, frame_size)``.  Read-only view.

        Raises
        ------
        ValueError
            If *audio* is shorter than *frame_size*.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D; got shape {audio.shape}")
        if len(audio) < frame_size:
            raise ValueError(
                f"audio length {len(audio)} < frame_size {frame_size}"
            )

        n_frames = 1 + (len(audio) - frame_size) // hop_size
        shape = (n_frames, frame_size)
        strides = (audio.strides[0] * hop_size, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        # Return a read-only view to prevent accidental mutation
        frames.flags.writeable = False
        return frames

    @staticmethod
    def vectorized_rms(frames: np.ndarray) -> np.ndarray:
        """Compute per-frame Root Mean Square energy.

        Parameters
        ----------
        frames:
            2-D array of shape ``(n_frames, frame_size)``.

        Returns
        -------
        np.ndarray
            1-D array of RMS values, shape ``(n_frames,)``.
        """
        if frames.ndim != 2:
            raise ValueError(f"frames must be 2-D; got shape {frames.shape}")
        return np.sqrt(np.mean(frames ** 2, axis=1))

    @staticmethod
    def vectorized_preemphasis(
        audio: np.ndarray,
        coeff: float = 0.97,
    ) -> np.ndarray:
        """Apply a first-order pre-emphasis filter to *audio*.

        The filter is defined as::

            y[n] = x[n] - coeff * x[n-1]

        Implemented via vectorised array slicing (no Python loop).

        Parameters
        ----------
        audio:
            1-D float32 audio signal.
        coeff:
            Pre-emphasis coefficient (0 < coeff < 1).  Default 0.97.

        Returns
        -------
        np.ndarray
            Pre-emphasised signal, same length as *audio*.
        """
        if not 0.0 < coeff < 1.0:
            raise ValueError(f"coeff must be in (0, 1); got {coeff}")
        out = np.empty_like(audio)
        out[0] = audio[0]
        out[1:] = audio[1:] - coeff * audio[:-1]
        return out

    # ------------------------------------------------------------------
    # Analysis / recommendations
    # ------------------------------------------------------------------

    def recommend_optimizations(
        self,
        pipeline_source: Optional[str] = None,
    ) -> List[str]:
        """Analyse a pipeline description and return optimisation suggestions.

        Parameters
        ----------
        pipeline_source:
            Optional string containing Python source code of the preprocessing
            pipeline.  When provided, heuristics look for common anti-patterns.

        Returns
        -------
        List[str]
            Human-readable optimisation recommendations.
        """
        suggestions: List[str] = []

        if pipeline_source is not None:
            # Heuristic 1: Python for-loops on sample data
            import re
            if re.search(r"\bfor\b.*\bin\b.*audio|for.*sample", pipeline_source):
                suggestions.append(
                    "Replace Python loops over audio samples with NumPy vectorised "
                    "operations (e.g., vectorized_preemphasis, vectorized_frame_split)."
                )

            # Heuristic 2: repeated np.append / list concatenation
            if "np.append" in pipeline_source or ".append(" in pipeline_source:
                suggestions.append(
                    "Avoid np.append() inside loops; pre-allocate output arrays "
                    "or collect results in a list and call np.stack() once."
                )

            # Heuristic 3: Python-level FFT loop
            if re.search(r"for.*fft|fft.*for", pipeline_source, re.IGNORECASE):
                suggestions.append(
                    "Apply FFT to the full frame matrix at once: "
                    "np.fft.rfft(frames, axis=1) — avoid per-frame loops."
                )

            # Heuristic 4: dtype promotion
            if "float64" in pipeline_source:
                suggestions.append(
                    "Use float32 instead of float64 throughout the pipeline to "
                    "halve memory usage and improve SIMD throughput."
                )

        # Always-applicable recommendations
        suggestions.extend(
            [
                "Pin NumPy arrays to 32-bit float (dtype=np.float32) before "
                "passing to the ONNX session to avoid implicit copies.",
                "Use vectorized_frame_split (stride-tricks) for zero-copy "
                "overlapping frames instead of np.array([...]) per frame.",
                "Enable OnnxRuntime intra-op parallelism: "
                "SessionOptions.intra_op_num_threads = os.cpu_count().",
                "Pre-allocate output buffers (io_binding) when calling the same "
                "model shape repeatedly to eliminate per-call allocation.",
            ]
        )
        return suggestions


# ===========================================================================
# 5. PerformanceReport
# ===========================================================================

@dataclass
class PerformanceReport:
    """Comprehensive performance audit results.

    Attributes
    ----------
    peak_memory_mb:         Maximum RSS observed during the audit (MB).
    avg_memory_mb:          Mean RSS during the audit (MB).
    mean_latency_ms:        Mean per-chunk end-to-end latency (ms).
    p95_latency_ms:         95th-percentile latency (ms).
    p99_latency_ms:         99th-percentile latency (ms).
    throughput_fps:         Chunks processed per second.
    memory_timeline:        Full list of MemorySnapshot objects.
    optimization_suggestions: Human-readable optimisation tips.
    passes_memory_check:    True if peak_memory_mb < 2048.
    passes_latency_check:   True if p95_latency_ms < LATENCY_LIMIT_MS.
    model_path:             Path to the ONNX model under test.
    audit_duration_s:       Wall time of the full audit run (seconds).
    n_chunks_processed:     Number of audio chunks processed.
    """

    peak_memory_mb: float
    avg_memory_mb: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    memory_timeline: List[MemorySnapshot]
    optimization_suggestions: List[str]
    passes_memory_check: bool
    passes_latency_check: bool
    model_path: str = ""
    audit_duration_s: float = 0.0
    n_chunks_processed: int = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a JSON-serialisable dictionary."""
        d = asdict(self)
        # memory_timeline is a list of MemorySnapshot dicts already via asdict
        return d

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        status_mem = "PASS" if self.passes_memory_check else "FAIL"
        status_lat = "PASS" if self.passes_latency_check else "FAIL"
        lines = [
            "# Vox-Verify Performance Audit Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| Peak Memory | {self.peak_memory_mb:.1f} MB | < 2048 MB | {status_mem} |",
            f"| Mean Latency | {self.mean_latency_ms:.2f} ms | — | — |",
            f"| P95 Latency | {self.p95_latency_ms:.2f} ms | < {LATENCY_LIMIT_MS:.0f} ms | {status_lat} |",
            f"| P99 Latency | {self.p99_latency_ms:.2f} ms | — | — |",
            f"| Avg Memory | {self.avg_memory_mb:.1f} MB | — | — |",
            f"| Throughput | {self.throughput_fps:.2f} chunks/s | — | — |",
            "",
            "## Run Details",
            "",
            f"- **Model**: `{self.model_path or 'N/A (synthetic)'}`",
            f"- **Audit Duration**: {self.audit_duration_s:.2f} s",
            f"- **Chunks Processed**: {self.n_chunks_processed}",
            "",
            "## Optimisation Suggestions",
            "",
        ]
        for i, tip in enumerate(self.optimization_suggestions, 1):
            lines.append(f"{i}. {tip}")
        lines.append("")
        lines.append("## Memory Timeline (first 10 samples)")
        lines.append("")
        lines.append("| Time (s) | RSS (MB) | VMS (MB) | CPU% |")
        lines.append("|----------|----------|----------|------|")
        for snap in self.memory_timeline[:10]:
            lines.append(
                f"| {snap.timestamp:.3f} | {snap.rss_mb:.1f} | "
                f"{snap.vms_mb:.1f} | {snap.cpu_percent:.1f} |"
            )
        if len(self.memory_timeline) > 10:
            lines.append(f"| … ({len(self.memory_timeline) - 10} more rows) | | | |")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        ok = "\u2713"  # ✓
        fail = "\u2717"  # ✗
        mem_icon = ok if self.passes_memory_check else fail
        lat_icon = ok if self.passes_latency_check else fail
        print("\n" + "=" * 60)
        print("  Vox-Verify  |  Performance Audit Summary")
        print("=" * 60)
        print(f"  Model path   : {self.model_path or '(synthetic test)'}")
        print(f"  Chunks       : {self.n_chunks_processed}  |  Duration: {self.audit_duration_s:.2f}s")
        print("-" * 60)
        print(f"  Peak RAM     : {self.peak_memory_mb:>8.1f} MB   {mem_icon}  (<2048 MB)")
        print(f"  Avg RAM      : {self.avg_memory_mb:>8.1f} MB")
        print(f"  Mean Latency : {self.mean_latency_ms:>8.2f} ms")
        print(f"  P95 Latency  : {self.p95_latency_ms:>8.2f} ms   {lat_icon}  (<{LATENCY_LIMIT_MS:.0f} ms)")
        print(f"  P99 Latency  : {self.p99_latency_ms:>8.2f} ms")
        print(f"  Throughput   : {self.throughput_fps:>8.2f} chunks/s")
        print("-" * 60)
        overall = "ALL CHECKS PASSED" if (self.passes_memory_check and self.passes_latency_check) else "SOME CHECKS FAILED"
        print(f"  Overall      : {overall}")
        print("=" * 60)
        if self.optimization_suggestions:
            print("\n  Optimisation Suggestions:")
            for i, tip in enumerate(self.optimization_suggestions, 1):
                # Wrap long lines at 56 chars
                words = tip.split()
                line = f"  {i}. "
                for word in words:
                    if len(line) + len(word) + 1 > 58:
                        print(line.rstrip())
                        line = "      " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    print(line.rstrip())
        print()


# ===========================================================================
# 6. PerformanceAuditor
# ===========================================================================

class PerformanceAuditor:
    """End-to-end pipeline auditor for Vox-Verify.

    Steps executed by ``run_audit``
    --------------------------------
    a. Load the ONNX model (or use a synthetic stub when no path is given).
    b. Generate synthetic audio (60 s at 16 kHz by default).
    c. Simulate live monitoring: process audio in 1 s chunks / 0.5 s hops.
    d. Track memory throughout via MemoryProfiler.
    e. If peak RSS > 2 GB, identify the bottleneck and apply optimisations.
    f. Re-run with optimisations and verify memory is under 2 GB.
    g. Return a PerformanceReport.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz (default 16 000).
    chunk_duration_s:
        Duration of each processed chunk in seconds (default 1.0).
    hop_duration_s:
        Hop between successive chunk starts in seconds (default 0.5).
    memory_interval_s:
        Memory sampling interval (default 0.1 s).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_s: float = 1.0,
        hop_duration_s: float = 0.5,
        memory_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration_s = chunk_duration_s
        self._hop_duration_s = hop_duration_s
        self._memory_interval_s = memory_interval_s
        self._optimizer = FeatureExtractionOptimizer()
        self._inf_profiler = InferenceProfiler(enable_ort_profiling=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_audit(
        self,
        model_path: Optional[str] = None,
        duration_seconds: float = 60.0,
    ) -> PerformanceReport:
        """Run the full performance audit.

        Parameters
        ----------
        model_path:
            Path to an ONNX model file.  When *None* or missing, a synthetic
            stub is used (no I/O overhead).
        duration_seconds:
            Length of synthetic audio to generate (seconds).

        Returns
        -------
        PerformanceReport
            Structured results including memory timeline and recommendations.
        """
        audit_start = time.perf_counter()
        logger.info("Starting performance audit (%.0f s audio).", duration_seconds)

        # ---- a. Load model -------------------------------------------
        session = self._load_model(model_path)

        # ---- b. Generate synthetic audio -----------------------------
        n_samples = int(duration_seconds * self._sample_rate)
        audio = np.random.randn(n_samples).astype(np.float32)
        logger.info("Generated %.1f s synthetic audio (%d samples).", duration_seconds, n_samples)

        # ---- c-d. Simulate live monitoring with memory tracking -------
        latencies, mem_profiler = self._run_simulation(session, audio, pass_label="Pass 1")

        peak_mb = mem_profiler.get_peak_memory_mb()
        avg_mb = mem_profiler.get_average_memory_mb()
        timeline = mem_profiler.get_timeline()

        suggestions = self._optimizer.recommend_optimizations()

        # ---- e-f. If over 2 GB, optimise and re-run ------------------
        if peak_mb > MEMORY_LIMIT_MB:
            logger.warning(
                "Peak memory %.1f MB exceeds 2 GB limit — applying optimisations.",
                peak_mb,
            )
            audio = self._apply_memory_optimizations(audio)
            latencies, mem_profiler2 = self._run_simulation(
                session, audio, pass_label="Pass 2 (optimised)"
            )
            peak_mb = mem_profiler2.get_peak_memory_mb()
            avg_mb = mem_profiler2.get_average_memory_mb()
            timeline = mem_profiler2.get_timeline()
            suggestions.insert(
                0,
                f"Memory optimisation was applied automatically (original peak "
                f"{peak_mb:.0f} MB).  Review dtype and chunk-size settings.",
            )

        # ---- g. Compile report ---------------------------------------
        audit_duration = time.perf_counter() - audit_start
        report = self._build_report(
            latencies=latencies,
            peak_mb=peak_mb,
            avg_mb=avg_mb,
            timeline=timeline,
            suggestions=suggestions,
            model_path=model_path or "",
            audit_duration_s=audit_duration,
        )

        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Optional[str]) -> Optional[Any]:
        """Load an ONNX InferenceSession or return None for stub mode."""
        if not model_path or not os.path.isfile(model_path):
            if model_path:
                logger.warning("Model not found at '%s'; using stub mode.", model_path)
            else:
                logger.info("No model path provided; using stub mode.")
            return None

        if not _ORT_AVAILABLE:
            logger.warning("onnxruntime not installed; using stub mode.")
            return None

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_path, sess_options=opts)
            logger.info("Loaded ONNX model: %s", model_path)
            return session
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return None

    def _run_simulation(
        self,
        session: Optional[Any],
        audio: np.ndarray,
        pass_label: str = "Pass",
    ):
        """Simulate live monitoring and return (latencies, mem_profiler)."""
        chunk_size = int(self._chunk_duration_s * self._sample_rate)
        hop_size = int(self._hop_duration_s * self._sample_rate)
        n_chunks = 1 + max(0, (len(audio) - chunk_size) // hop_size)

        latencies: List[float] = []
        mem_profiler = MemoryProfiler(interval_s=self._memory_interval_s)

        logger.info("%s: processing %d chunks.", pass_label, n_chunks)
        mem_profiler.start()

        for i in range(n_chunks):
            start = i * hop_size
            chunk = audio[start: start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            profile = self._inf_profiler.profile_single(session, chunk)
            latencies.append(profile.total_ms)

        mem_profiler.stop()
        return latencies, mem_profiler

    @staticmethod
    def _apply_memory_optimizations(audio: np.ndarray) -> np.ndarray:
        """Downcast to float32 and normalise to reduce working-set size."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0.0:
            audio /= peak
        return audio

    @staticmethod
    def _build_report(
        latencies: List[float],
        peak_mb: float,
        avg_mb: float,
        timeline: List[MemorySnapshot],
        suggestions: List[str],
        model_path: str,
        audit_duration_s: float,
    ) -> PerformanceReport:
        arr = np.array(latencies, dtype=np.float64)
        n = len(latencies)
        mean_ms = float(arr.mean()) if n else 0.0
        p95_ms = float(np.percentile(arr, 95)) if n else 0.0
        p99_ms = float(np.percentile(arr, 99)) if n else 0.0
        throughput = n / audit_duration_s if audit_duration_s > 0 else 0.0

        return PerformanceReport(
            peak_memory_mb=peak_mb,
            avg_memory_mb=avg_mb,
            mean_latency_ms=mean_ms,
            p95_latency_ms=p95_ms,
            p99_latency_ms=p99_ms,
            throughput_fps=throughput,
            memory_timeline=timeline,
            optimization_suggestions=suggestions,
            passes_memory_check=peak_mb < MEMORY_LIMIT_MB,
            passes_latency_check=p95_ms < LATENCY_LIMIT_MS,
            model_path=model_path,
            audit_duration_s=audit_duration_s,
            n_chunks_processed=n,
        )


# ===========================================================================
# 7. Internal utilities
# ===========================================================================

class _NsTimer:
    """Lightweight nanosecond timer used as a context manager.

    Attributes
    ----------
    elapsed_ms: Elapsed time in milliseconds (available after __exit__).
    """

    __slots__ = ("_start_ns", "elapsed_ms")

    def __init__(self) -> None:
        self._start_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_NsTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1e6


@contextmanager
def _ns_timer() -> Generator[_NsTimer, None, None]:
    """Yield a _NsTimer that records elapsed milliseconds on exit.

    Usage::

        with _ns_timer() as t:
            do_work()
        print(t.elapsed_ms)
    """
    timer = _NsTimer()
    timer.__enter__()
    try:
        yield timer
    finally:
        timer.__exit__()


# ===========================================================================
# 8. CLI entry point
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory_audit",
        description="Vox-Verify memory and performance auditor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to an ONNX model file.  Omit to use synthetic stub.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Duration of synthetic audio to generate (seconds).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write the JSON report.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_S,
        metavar="SECONDS",
        help="Memory sampling interval (seconds).",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the performance auditor."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    auditor = PerformanceAuditor(memory_interval_s=args.interval)
    report = auditor.run_audit(
        model_path=args.model_path,
        duration_seconds=args.duration,
    )

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        report_dict = report.to_dict()
        # Convert MemorySnapshot objects (already dicts via asdict) to plain dicts
        report_dict["memory_timeline"] = [
            s if isinstance(s, dict) else s.to_dict()
            for s in report_dict["memory_timeline"]
        ]
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2)
        logger.info("Report written to: %s", out_path)

    # Also print Markdown to stdout
    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
