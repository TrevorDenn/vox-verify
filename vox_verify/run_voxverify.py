#!/usr/bin/env python3
"""Vox-Verify Local — Main entry point.

Usage:
    # Launch the Streamlit dashboard
    python run_voxverify.py ui

    # Run the adversarial stress test
    python run_voxverify.py test --model weights/aasist_tiny.onnx

    # Export and quantize the model
    python run_voxverify.py quantize --output weights/

    # Run the performance audit
    python run_voxverify.py audit --model weights/aasist_tiny.onnx

    # Run headless monitoring (no UI)
    python run_voxverify.py monitor --model weights/aasist_tiny.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def cmd_ui(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard = BASE_DIR / "ui" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard),
           "--server.headless", "true", "--server.port", str(args.port)]
    print(f"Launching Vox-Verify dashboard on port {args.port}…")
    subprocess.run(cmd, cwd=str(BASE_DIR))


def cmd_test(args: argparse.Namespace) -> None:
    """Run the adversarial stress test suite."""
    sys.path.insert(0, str(BASE_DIR.parent))
    from vox_verify.tests.adversarial_stress_test import AdversarialTestRunner

    output_dir = Path(args.output) if args.output else BASE_DIR / "test_output"
    runner = AdversarialTestRunner(
        model_path=args.model,
        output_dir=str(output_dir),
        sample_rate=16000,
        seed=42,
    )
    report = runner.run_all()

    print(f"\n{'=' * 50}")
    print(f"  ADVERSARIAL STRESS TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Total:     {report.total_tests}")
    print(f"  Passed:    {report.passed}")
    print(f"  Failed:    {report.failed}")
    print(f"  Pass Rate: {report.pass_rate:.1%}")
    print(f"  Mean Lat:  {report.latency_mean_ms:.2f} ms")
    print(f"  P95 Lat:   {report.latency_p95_ms:.2f} ms")
    print(f"{'=' * 50}")

    if report.pass_rate >= 0.80:
        print("  ✓ PASS — Stress test threshold met (≥80%)")
    else:
        print("  ✗ FAIL — Stress test threshold NOT met (<80%)")

    if report.latency_mean_ms < 100:
        print(f"  ✓ PASS — Latency target met (<100ms, got {report.latency_mean_ms:.1f}ms)")
    else:
        print(f"  ✗ FAIL — Latency target NOT met (>100ms, got {report.latency_mean_ms:.1f}ms)")


def cmd_quantize(args: argparse.Namespace) -> None:
    """Export and quantize the model."""
    import torch
    sys.path.insert(0, str(BASE_DIR.parent))
    from vox_verify.models.aasist_tiny import AASISTTiny

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading AASIST-Tiny model…")
    model = AASISTTiny()
    model.eval()

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # Save PyTorch checkpoint
    pt_path = output / "aasist_tiny.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"  PyTorch: {pt_path} ({pt_path.stat().st_size / 1024:.1f} KB)")

    # ONNX export
    onnx_path = output / "aasist_tiny.onnx"
    x = torch.randn(1, 1, 16000)
    torch.onnx.export(
        model, x, str(onnx_path), opset_version=18,
        input_names=["audio"], output_names=["logits"],
        dynamic_axes={"audio": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"  ONNX FP32: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # ONNX Runtime graph optimization
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt_path = output / "aasist_tiny_optimized.onnx"
    so.optimized_model_filepath = str(opt_path)
    ort.InferenceSession(str(onnx_path), so)
    print(f"  ONNX Optimized: {opt_path} ({opt_path.stat().st_size / 1024:.1f} KB)")

    # PyTorch dynamic INT8
    model_q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    q_path = output / "aasist_tiny_int8.pt"
    torch.save(model_q.state_dict(), q_path)
    print(f"  PyTorch INT8: {q_path} ({q_path.stat().st_size / 1024:.1f} KB)")

    # Benchmark all variants
    import numpy as np
    inp = np.random.randn(1, 1, 16000).astype(np.float32)
    print("\nBenchmark (100 iterations each):")

    for name, path in [("FP32", onnx_path), ("Optimized", opt_path)]:
        if path.exists():
            sess = ort.InferenceSession(str(path))
            for _ in range(10):
                sess.run(None, {"audio": inp})
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {"audio": inp})
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            print(f"  {name:12s}: Mean={sum(times)/len(times):.2f}ms  P95={times[94]:.2f}ms")


def cmd_audit(args: argparse.Namespace) -> None:
    """Run the performance memory audit."""
    sys.path.insert(0, str(BASE_DIR.parent))
    from vox_verify.profiler.memory_audit import PerformanceAuditor

    auditor = PerformanceAuditor()
    report = auditor.run_audit(
        model_path=args.model,
        duration_seconds=args.duration,
    )
    report.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to {args.output}")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Headless monitoring mode (console output only)."""
    sys.path.insert(0, str(BASE_DIR.parent))
    from vox_verify.stream.buffer_engine import BufferEngine, EngineConfig

    config = EngineConfig(
        model_path=args.model,
        device_index=args.device,
        sensitivity_threshold=args.threshold,
    )

    def on_detection(result):
        status = "🔴 SPOOF" if result.is_spoof else "🟢 SAFE"
        print(f"[{result.timestamp}] {status}  "
              f"bonafide={result.bonafide_score:.3f}  "
              f"spoof={result.spoof_score:.3f}  "
              f"latency={result.latency_ms:.1f}ms")

    engine = BufferEngine(config)
    engine.on_detection = on_detection

    print("Starting Vox-Verify headless monitor…")
    print("Press Ctrl+C to stop.\n")

    try:
        engine.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voxverify",
        description="Vox-Verify Local — Audio Deepfake Detector",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # UI
    p_ui = subs.add_parser("ui", help="Launch the Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501)

    # Test
    p_test = subs.add_parser("test", help="Run adversarial stress tests")
    p_test.add_argument("--model", required=True, help="Path to ONNX model")
    p_test.add_argument("--output", help="Output directory for test results")

    # Quantize
    p_quant = subs.add_parser("quantize", help="Export and quantize the model")
    p_quant.add_argument("--output", required=True, help="Output directory")
    p_quant.add_argument("--checkpoint", help="PyTorch checkpoint to load")

    # Audit
    p_audit = subs.add_parser("audit", help="Run performance memory audit")
    p_audit.add_argument("--model", required=True, help="Path to ONNX model")
    p_audit.add_argument("--duration", type=int, default=60)
    p_audit.add_argument("--output", help="JSON report output path")

    # Monitor
    p_mon = subs.add_parser("monitor", help="Headless monitoring mode")
    p_mon.add_argument("--model", required=True, help="Path to ONNX model")
    p_mon.add_argument("--device", type=int, default=None)
    p_mon.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    dispatch = {
        "ui": cmd_ui,
        "test": cmd_test,
        "quantize": cmd_quantize,
        "audit": cmd_audit,
        "monitor": cmd_monitor,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
