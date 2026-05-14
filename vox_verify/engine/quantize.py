"""
vox_verify.engine.quantize
==========================
ONNX export and quantization engine for the AASIST-Tiny deepfake detection model.

Supports:
  - PyTorch → ONNX export (opset 17, dynamic batch axis)
  - FP16 weight conversion
  - INT8 dynamic quantization  (no calibration data required)
  - INT8 static quantization   (calibration data required)
  - Inference benchmarking with latency and throughput metrics

Typical usage
-------------
  # From Python
  from vox_verify.engine.quantize import export_to_onnx, quantize_int8_dynamic, benchmark_inference

  export_to_onnx(model, "model.onnx")
  quantize_int8_dynamic("model.onnx", "model_int8.onnx")
  benchmark_inference(["model.onnx", "model_int8.onnx"])

  # From CLI
  python quantize.py --model_path checkpoint.pt --output_dir ./onnx_models --quantize all
"""

from __future__ import annotations

import abc
import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so the module can be imported
# even when only onnxruntime is available (e.g. in an inference-only container).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # Hz
DUMMY_DURATION_S: float = 1.0      # seconds used for the export dummy input
DUMMY_INPUT_SHAPE: Tuple[int, int, int] = (1, 1, int(SAMPLE_RATE * DUMMY_DURATION_S))

# ── Benchmark defaults ─────────────────────────────────────────────────────
BENCH_WARMUP_ITERS: int = 10
BENCH_MEASURE_ITERS: int = 100


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _require_torch():
    """Import and return torch, raising ImportError with an actionable message."""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for ONNX export. "
            "Install it with: pip install torch"
        ) from exc


def _require_onnx():
    """Import and return onnx, raising ImportError with an actionable message."""
    try:
        import onnx  # noqa: PLC0415
        return onnx
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required. "
            "Install it with: pip install onnx"
        ) from exc


def _require_ort_quantization():
    """Import and return the onnxruntime.quantization module."""
    try:
        import onnxruntime.quantization as ortq  # noqa: PLC0415
        return ortq
    except ImportError as exc:
        raise ImportError(
            "onnxruntime with quantization support is required. "
            "Install it with: pip install onnxruntime"
        ) from exc


def _require_ort():
    """Import and return onnxruntime."""
    try:
        import onnxruntime as ort  # noqa: PLC0415
        return ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. "
            "Install it with: pip install onnxruntime"
        ) from exc


def _file_size_mb(path: Union[str, Path]) -> float:
    """Return the size of *path* in megabytes."""
    return os.path.getsize(path) / (1024 ** 2)


# ---------------------------------------------------------------------------
# 1. PyTorch → ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model,  # torch.nn.Module
    output_path: Union[str, Path],
    *,
    opset_version: int = 17,
    verify: bool = True,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Creates a dummy audio input of shape ``(1, 1, 16000)`` (one second of
    mono 16 kHz audio) and traces the model through ``torch.onnx.export``.
    The batch dimension is kept dynamic so the exported model accepts
    variable-length batches at runtime.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` instance (AASIST-Tiny or compatible).
    output_path:
        Destination ``.onnx`` file path.
    opset_version:
        ONNX opset version. Defaults to 17 (recommended for modern ops).
    verify:
        When ``True`` (default), runs ``onnx.checker.check_model`` on the
        exported file and raises ``ValueError`` if the model is invalid.
    input_names:
        Override the default ``["audio_input"]`` input tensor names.
    output_names:
        Override the default ``["logits"]`` output tensor names.

    Returns
    -------
    Path
        Resolved path to the exported ``.onnx`` file.

    Raises
    ------
    ImportError
        If ``torch`` or ``onnx`` are not installed.
    ValueError
        If the exported model fails ONNX validation.
    RuntimeError
        If the export itself fails (e.g. unsupported op).
    """
    torch = _require_torch()
    onnx = _require_onnx()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = input_names or ["audio_input"]
    output_names = output_names or ["logits"]

    # Dynamic axes: allow the batch dimension (axis 0) to vary at runtime.
    dynamic_axes: Dict[str, Dict[int, str]] = {
        name: {0: "batch_size"} for name in input_names + output_names
    }

    logger.info("Preparing dummy input %s for ONNX export …", DUMMY_INPUT_SHAPE)
    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE)

    # Ensure the model is in eval mode and on CPU for a deterministic export.
    model.eval()
    model_device = next(model.parameters(), torch.zeros(1)).device
    export_model = model.cpu()
    export_input = dummy_input.cpu()

    logger.info("Exporting to ONNX (opset %d): %s", opset_version, output_path)
    try:
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                export_input,
                str(output_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc
    finally:
        # Move model back to original device if needed.
        if str(model_device) != "cpu":
            model.to(model_device)

    size_mb = _file_size_mb(output_path)
    logger.info("Exported ONNX model (%.2f MB): %s", size_mb, output_path)

    if verify:
        logger.info("Verifying ONNX model …")
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed.")
        except onnx.checker.ValidationError as exc:
            raise ValueError(f"Exported ONNX model is invalid: {exc}") from exc

    return output_path


# ---------------------------------------------------------------------------
# 2. FP16 quantization
# ---------------------------------------------------------------------------

def quantize_fp16(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    keep_io_types: bool = True,
) -> Path:
    """Convert an ONNX model's float32 weights to float16.

    Uses the ``onnxruntime.transformers.float16`` converter which is bundled
    with ``onnxruntime`` and handles edge cases such as ops that must remain
    in float32 (e.g. Resize, LayerNorm inputs).

    Parameters
    ----------
    model_path:
        Path to the source float32 ``.onnx`` model.
    output_path:
        Destination path for the float16 ``.onnx`` model.
    keep_io_types:
        When ``True`` (default), the model's input and output tensors remain
        float32 so callers can pass standard float32 buffers. Only internal
        weights and activations are converted to float16.

    Returns
    -------
    Path
        Resolved path to the float16 model.

    Raises
    ------
    ImportError
        If ``onnx`` or ``onnxruntime`` are not installed.
    FileNotFoundError
        If *model_path* does not exist.
    """
    onnx = _require_onnx()

    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ONNX model for FP16 conversion: %s", model_path)
    onnx_model = onnx.load(str(model_path))

    # Prefer the onnxruntime.transformers converter (already a dependency).
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16  # noqa: PLC0415
        logger.info("Using onnxruntime.transformers float16 converter.")
        fp16_model = convert_float_to_float16(
            onnx_model,
            keep_io_types=keep_io_types,
            disable_shape_infer=False,
        )
    except ImportError:
        # Fallback: onnxconverter_common (must be separately installed).
        try:
            from onnxconverter_common import float16 as _f16  # noqa: PLC0415
            logger.info("Using onnxconverter_common float16 converter.")
            fp16_model = _f16.convert_float_to_float16(
                onnx_model,
                keep_io_types=keep_io_types,
            )
        except ImportError as exc:
            raise ImportError(
                "FP16 conversion requires either onnxruntime (>=1.12) or "
                "onnxconverter_common. Install with: pip install onnxconverter-common"
            ) from exc

    onnx.save(fp16_model, str(output_path))

    src_mb = _file_size_mb(model_path)
    dst_mb = _file_size_mb(output_path)
    reduction_pct = (1 - dst_mb / src_mb) * 100 if src_mb > 0 else 0.0

    logger.info(
        "FP16 model saved: %s  (%.2f MB → %.2f MB, %.1f%% reduction)",
        output_path, src_mb, dst_mb, reduction_pct,
    )
    print(
        f"[FP16]  {model_path.name} → {output_path.name}  "
        f"{src_mb:.2f} MB → {dst_mb:.2f} MB  ({reduction_pct:.1f}% smaller)"
    )
    return output_path


# ---------------------------------------------------------------------------
# 3. INT8 dynamic quantization
# ---------------------------------------------------------------------------

def quantize_int8_dynamic(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    op_types_to_quantize: Optional[List[str]] = None,
    per_channel: bool = True,
) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model.

    Dynamic quantization computes activation scales at runtime, so no
    calibration data is required. Weight tensors are statically quantized
    to INT8; activations are quantized dynamically during inference.

    This method targets Conv and MatMul operators with per-channel weight
    quantization for best accuracy on CPUs without VNNI support.

    Parameters
    ----------
    model_path:
        Path to the source (float32) ``.onnx`` model.
    output_path:
        Destination path for the INT8 ``.onnx`` model.
    op_types_to_quantize:
        List of op types to quantize. Defaults to ``["Conv", "MatMul"]``.
    per_channel:
        Apply per-channel (rather than per-tensor) quantization to weight
        tensors. Improves accuracy for Conv layers. Defaults to ``True``.

    Returns
    -------
    Path
        Resolved path to the quantized model.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If *model_path* does not exist.
    """
    ortq = _require_ort_quantization()

    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    op_types = op_types_to_quantize or ["Conv", "MatMul"]
    logger.info(
        "Applying INT8 dynamic quantization (per_channel=%s, ops=%s): %s",
        per_channel, op_types, model_path,
    )

    ortq.quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        op_types_to_quantize=op_types,
        per_channel=per_channel,
        reduce_range=False,          # full 8-bit range; set True for non-VNNI
        weight_type=ortq.QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )

    src_mb = _file_size_mb(model_path)
    dst_mb = _file_size_mb(output_path)
    reduction_pct = (1 - dst_mb / src_mb) * 100 if src_mb > 0 else 0.0

    logger.info(
        "INT8-dynamic model saved: %s  (%.2f MB → %.2f MB, %.1f%% reduction)",
        output_path, src_mb, dst_mb, reduction_pct,
    )
    print(
        f"[INT8-dynamic]  {model_path.name} → {output_path.name}  "
        f"{src_mb:.2f} MB → {dst_mb:.2f} MB  ({reduction_pct:.1f}% smaller)"
    )
    return output_path


# ---------------------------------------------------------------------------
# 4. INT8 static quantization
# ---------------------------------------------------------------------------

class AudioCalibrationDataReader:
    """Feeds random (or real) audio chunks to the ONNX calibrator.

    Implements the ``onnxruntime.quantization.CalibrationDataReader`` ABC
    so it can be passed directly to :func:`quantize_int8_static`.

    Parameters
    ----------
    input_name:
        The ONNX model's audio input tensor name (e.g. ``"audio_input"``).
    data_loader:
        An optional iterable that yields ``np.ndarray`` chunks of shape
        ``(batch, 1, samples)``.  When ``None``, random synthetic data is
        generated using *n_samples* and *sample_length*.
    n_samples:
        Number of synthetic calibration samples to generate when no
        ``data_loader`` is provided.
    sample_length:
        Length (in audio samples) of each synthetic chunk.  Defaults to
        16 000 (one second at 16 kHz).
    seed:
        Random seed for reproducible synthetic data.
    """

    def __init__(
        self,
        input_name: str = "audio_input",
        data_loader: Optional[Iterable[np.ndarray]] = None,
        *,
        n_samples: int = 100,
        sample_length: int = SAMPLE_RATE,
        seed: int = 42,
    ) -> None:
        self.input_name = input_name
        self._rng = np.random.default_rng(seed)

        if data_loader is not None:
            self._data: List[np.ndarray] = list(data_loader)
        else:
            logger.info(
                "No calibration data provided; generating %d synthetic samples "
                "(length=%d).", n_samples, sample_length,
            )
            self._data = [
                self._rng.standard_normal((1, 1, sample_length)).astype(np.float32)
                for _ in range(n_samples)
            ]

        self._index: int = 0
        logger.info("CalibrationDataReader ready with %d samples.", len(self._data))

    # ------------------------------------------------------------------
    # CalibrationDataReader protocol
    # ------------------------------------------------------------------

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the next calibration batch or ``None`` when exhausted."""
        if self._index >= len(self._data):
            return None
        chunk = self._data[self._index]
        self._index += 1
        # Ensure float32 and correct shape (batch, channels, samples).
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, np.newaxis, :]  # add batch + channel dims
        elif chunk.ndim == 2:
            chunk = chunk[np.newaxis, :]              # add batch dim
        return {self.input_name: chunk.astype(np.float32)}

    def rewind(self) -> None:
        """Reset the iterator to the beginning (required by the ORT calibrator)."""
        self._index = 0


# Make AudioCalibrationDataReader inherit from the real ABC when ORT is available.
# We do this at class-definition time so that isinstance checks pass.
try:
    from onnxruntime.quantization.calibrate import CalibrationDataReader as _OrtCDR  # noqa: PLC0415
    AudioCalibrationDataReader.__bases__ = (_OrtCDR,)
except Exception:  # onnx not yet importable — define a plain class
    pass


def quantize_int8_static(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_data_reader: "AudioCalibrationDataReader",
    *,
    op_types_to_quantize: Optional[List[str]] = None,
    per_channel: bool = True,
    quant_format: str = "QDQ",
) -> Path:
    """Apply INT8 static quantization using calibration data.

    Unlike dynamic quantization, static quantization pre-computes
    activation scales from representative calibration data.  This yields
    better performance at the cost of requiring a representative dataset.

    The QDQ (QuantizeLinear / DeQuantizeLinear) format is used by default
    because it produces the most portable and interoperable INT8 model.

    Parameters
    ----------
    model_path:
        Path to the source (float32) ``.onnx`` model.
    output_path:
        Destination path for the INT8 ``.onnx`` model.
    calibration_data_reader:
        An :class:`AudioCalibrationDataReader` (or any object implementing
        the ``CalibrationDataReader`` protocol) that yields input batches.
    op_types_to_quantize:
        List of op types to quantize. Defaults to ``["Conv", "MatMul"]``.
    per_channel:
        Apply per-channel weight quantization. Defaults to ``True``.
    quant_format:
        Either ``"QDQ"`` (default) or ``"QOperator"``.  ``"QDQ"`` inserts
        explicit Quantize/DeQuantize nodes; ``"QOperator"`` fuses them into
        operator kernels for potentially higher CPU throughput.

    Returns
    -------
    Path
        Resolved path to the quantized model.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If *model_path* does not exist.
    ValueError
        If an unknown *quant_format* is specified.
    """
    ortq = _require_ort_quantization()

    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map string format name to enum.
    _format_map = {
        "QDQ": ortq.QuantFormat.QDQ,
        "QOperator": ortq.QuantFormat.QOperator,
        "QOPERATOR": ortq.QuantFormat.QOperator,
    }
    if quant_format not in _format_map:
        raise ValueError(
            f"Unknown quant_format '{quant_format}'. Choose 'QDQ' or 'QOperator'."
        )
    ort_format = _format_map[quant_format]

    op_types = op_types_to_quantize or ["Conv", "MatMul"]
    logger.info(
        "Applying INT8 static quantization (format=%s, per_channel=%s, ops=%s): %s",
        quant_format, per_channel, op_types, model_path,
    )

    ortq.quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_data_reader,
        quant_format=ort_format,
        op_types_to_quantize=op_types,
        per_channel=per_channel,
        reduce_range=False,
        activation_type=ortq.QuantType.QInt8,
        weight_type=ortq.QuantType.QInt8,
        calibrate_method=ortq.CalibrationMethod.MinMax,
    )

    src_mb = _file_size_mb(model_path)
    dst_mb = _file_size_mb(output_path)
    reduction_pct = (1 - dst_mb / src_mb) * 100 if src_mb > 0 else 0.0

    logger.info(
        "INT8-static model saved: %s  (%.2f MB → %.2f MB, %.1f%% reduction)",
        output_path, src_mb, dst_mb, reduction_pct,
    )
    print(
        f"[INT8-static]  {model_path.name} → {output_path.name}  "
        f"{src_mb:.2f} MB → {dst_mb:.2f} MB  ({reduction_pct:.1f}% smaller)"
    )
    return output_path


# ---------------------------------------------------------------------------
# 5. Inference benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    model_paths: Sequence[Union[str, Path]],
    *,
    input_shape: Tuple[int, ...] = DUMMY_INPUT_SHAPE,
    warmup_iters: int = BENCH_WARMUP_ITERS,
    measure_iters: int = BENCH_MEASURE_ITERS,
    intra_op_num_threads: int = 1,
    providers: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """Benchmark one or more ONNX models and return latency statistics.

    Runs *measure_iters* forward passes with random input and reports mean,
    P95, P99 latency (ms) and throughput (inferences / second) for each
    model.  A comparison table is printed to stdout.

    Parameters
    ----------
    model_paths:
        Sequence of paths to ``.onnx`` files to benchmark.
    input_shape:
        Shape of the random input tensor. Defaults to ``(1, 1, 16000)``.
    warmup_iters:
        Number of un-timed warm-up inference passes. Defaults to 10.
    measure_iters:
        Number of timed inference passes. Defaults to 100.
    intra_op_num_threads:
        Number of threads for intra-operator parallelism.  Set to ``1``
        for latency benchmarking; increase for throughput benchmarking.
    providers:
        List of OnnxRuntime execution providers. Defaults to
        ``["CPUExecutionProvider"]``.

    Returns
    -------
    list of dict
        One dictionary per model with keys:
        ``model``, ``size_mb``, ``mean_ms``, ``p95_ms``, ``p99_ms``,
        ``throughput_ips``.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If any model path does not exist.
    """
    ort = _require_ort()

    providers = providers or ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_num_threads
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    rng = np.random.default_rng(0)
    dummy_input = rng.standard_normal(input_shape).astype(np.float32)

    results: List[Dict[str, float]] = []

    for raw_path in model_paths:
        model_path = Path(raw_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading session: %s", model_path)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Resolve input name from session metadata.
        input_name = session.get_inputs()[0].name

        # --- Warm-up ---
        for _ in range(warmup_iters):
            session.run(None, {input_name: dummy_input})

        # --- Measurement ---
        latencies_ms: List[float] = []
        t_start_total = time.perf_counter()
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
        t_end_total = time.perf_counter()

        arr = np.array(latencies_ms)
        mean_ms = float(np.mean(arr))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        total_s = t_end_total - t_start_total
        throughput = measure_iters / total_s

        size_mb = _file_size_mb(model_path)
        logger.info(
            "[%s] mean=%.2f ms  p95=%.2f ms  p99=%.2f ms  %.1f inf/s",
            model_path.name, mean_ms, p95_ms, p99_ms, throughput,
        )

        results.append(
            {
                "model": model_path.name,
                "size_mb": round(size_mb, 2),
                "mean_ms": round(mean_ms, 3),
                "p95_ms": round(p95_ms, 3),
                "p99_ms": round(p99_ms, 3),
                "throughput_ips": round(throughput, 2),
            }
        )

    _print_benchmark_table(results)
    return results


def _print_benchmark_table(results: List[Dict]) -> None:
    """Pretty-print a benchmark comparison table to stdout."""
    if not results:
        return

    col_widths = {
        "model":           max(len("Model"), max(len(r["model"]) for r in results)),
        "size_mb":         10,
        "mean_ms":         10,
        "p95_ms":          10,
        "p99_ms":          10,
        "throughput_ips":  14,
    }

    header = (
        f"{'Model':<{col_widths['model']}}  "
        f"{'Size (MB)':>{col_widths['size_mb']}}  "
        f"{'Mean (ms)':>{col_widths['mean_ms']}}  "
        f"{'P95 (ms)':>{col_widths['p95_ms']}}  "
        f"{'P99 (ms)':>{col_widths['p99_ms']}}  "
        f"{'Inf/s':>{col_widths['throughput_ips']}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  ONNX Inference Benchmark")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['model']:<{col_widths['model']}}  "
            f"{r['size_mb']:>{col_widths['size_mb']}.2f}  "
            f"{r['mean_ms']:>{col_widths['mean_ms']}.3f}  "
            f"{r['p95_ms']:>{col_widths['p95_ms']}.3f}  "
            f"{r['p99_ms']:>{col_widths['p99_ms']}.3f}  "
            f"{r['throughput_ips']:>{col_widths['throughput_ips']}.2f}"
        )

    print(sep)
    if len(results) > 1:
        baseline = results[0]
        for r in results[1:]:
            speedup = (
                baseline["mean_ms"] / r["mean_ms"]
                if r["mean_ms"] > 0 else float("inf")
            )
            size_ratio = (
                baseline["size_mb"] / r["size_mb"]
                if r["size_mb"] > 0 else float("inf")
            )
            print(
                f"  {r['model']} vs {baseline['model']}: "
                f"{speedup:.2f}x faster, {size_ratio:.2f}x smaller"
            )
    print()


# ---------------------------------------------------------------------------
# 6. Full pipeline helper
# ---------------------------------------------------------------------------

def run_full_pipeline(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    quantize_modes: Sequence[str],
    *,
    calibration_samples: int = 100,
) -> None:
    """Run the complete export → quantize → benchmark pipeline.

    Parameters
    ----------
    model_path:
        Path to a PyTorch checkpoint (``.pt`` / ``.pth``) **or** an already
        exported ``.onnx`` file.  When a ``.pt`` checkpoint is supplied the
        model is loaded via ``torch.load`` and exported first.
    output_dir:
        Directory where all ``.onnx`` artefacts are written.
    quantize_modes:
        One or more of ``"fp16"``, ``"int8-dynamic"``, ``"int8-static"``,
        or ``"all"`` (expands to all three).
    calibration_samples:
        Number of synthetic calibration samples for static INT8 quantization.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve quantize modes.
    all_modes = {"fp16", "int8-dynamic", "int8-static"}
    if "all" in quantize_modes:
        modes = all_modes
    else:
        modes = {m.lower() for m in quantize_modes} & all_modes
        unknown = {m.lower() for m in quantize_modes} - all_modes - {"all"}
        if unknown:
            logger.warning("Unknown quantize modes (ignored): %s", unknown)

    model_path = Path(model_path).resolve()

    # ── Step 1: Export or locate the base ONNX model ─────────────────────
    if model_path.suffix.lower() in {".pt", ".pth"}:
        torch = _require_torch()
        logger.info("Loading PyTorch checkpoint: %s", model_path)
        checkpoint = torch.load(str(model_path), map_location="cpu")

        # Support bare model objects, state-dict dicts, or {"model": ...} dicts.
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                pytorch_model = checkpoint["model"]
            else:
                # Assume it's a state dict — can't export without the arch.
                raise ValueError(
                    "Checkpoint is a state dict without architecture. "
                    "Pass a full model object or use a checkpoint that includes "
                    "the model under the key 'model'."
                )
        else:
            pytorch_model = checkpoint

        base_onnx = output_dir / "model.onnx"
        export_to_onnx(pytorch_model, base_onnx)
    elif model_path.suffix.lower() == ".onnx":
        base_onnx = model_path
        logger.info("Using existing ONNX model: %s", base_onnx)
    else:
        raise ValueError(
            f"Unsupported model file type: '{model_path.suffix}'. "
            "Expected .pt, .pth, or .onnx"
        )

    generated: List[Path] = [base_onnx]

    # ── Step 2: Quantize ──────────────────────────────────────────────────
    if "fp16" in modes:
        fp16_path = output_dir / "model_fp16.onnx"
        try:
            quantize_fp16(base_onnx, fp16_path)
            generated.append(fp16_path)
        except Exception as exc:
            logger.error("FP16 quantization failed: %s", exc)

    if "int8-dynamic" in modes:
        int8_dyn_path = output_dir / "model_int8_dynamic.onnx"
        try:
            quantize_int8_dynamic(base_onnx, int8_dyn_path)
            generated.append(int8_dyn_path)
        except Exception as exc:
            logger.error("INT8-dynamic quantization failed: %s", exc)

    if "int8-static" in modes:
        int8_static_path = output_dir / "model_int8_static.onnx"
        try:
            # Determine input name from the base ONNX model.
            ort = _require_ort()
            _sess = ort.InferenceSession(
                str(base_onnx), providers=["CPUExecutionProvider"]
            )
            input_name = _sess.get_inputs()[0].name

            calib_reader = AudioCalibrationDataReader(
                input_name=input_name,
                n_samples=calibration_samples,
            )
            quantize_int8_static(base_onnx, int8_static_path, calib_reader)
            generated.append(int8_static_path)
        except Exception as exc:
            logger.error("INT8-static quantization failed: %s", exc)

    # ── Step 3: Benchmark ─────────────────────────────────────────────────
    existing = [p for p in generated if p.exists()]
    if existing:
        try:
            benchmark_inference(existing)
        except Exception as exc:
            logger.error("Benchmarking failed: %s", exc)
    else:
        logger.warning("No models available for benchmarking.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantize",
        description=(
            "Export a PyTorch AASIST-Tiny model to ONNX and apply "
            "INT8/FP16 quantization for fast CPU inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help=(
            "Path to a PyTorch checkpoint (.pt/.pth) or an already-exported "
            "ONNX model (.onnx)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=Path("./onnx_models"),
        type=Path,
        help="Directory where quantized ONNX files are written.",
    )
    parser.add_argument(
        "--quantize",
        nargs="+",
        default=["all"],
        choices=["fp16", "int8-dynamic", "int8-static", "all"],
        metavar="MODE",
        help=(
            "Quantization mode(s) to apply. "
            "Choices: fp16, int8-dynamic, int8-static, all."
        ),
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=100,
        help="Number of synthetic calibration samples for INT8-static.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_full_pipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantize_modes=args.quantize,
        calibration_samples=args.calibration_samples,
    )
