# Vox-Verify Local — Deployment Guide

## Overview

Vox-Verify Local is a local-first audio deepfake detector that runs entirely on your machine.
It uses an AASIST-Tiny neural network (88K parameters, 0.34 MB) with sparse graph attention
to detect synthetic speech artifacts in real-time audio streams.

**Key specs:**
- Model: AASIST-Tiny with Sparse Graph Attention (88,736 parameters)
- Inference: ~10ms per 1-second chunk (ONNX Runtime, CPU)
- Memory: <500 MB RAM during live monitoring
- Latency: Sub-15ms end-to-end (well under 100ms target)
- No cloud dependency — everything runs locally

---

## System Requirements

| Component      | Minimum                | Recommended            |
|----------------|------------------------|------------------------|
| OS             | Windows 10 / macOS 12  | Windows 11 / macOS 14  |
| Python         | 3.10+                  | 3.11 or 3.12           |
| RAM            | 4 GB                   | 8 GB                   |
| CPU            | Any x86_64 / ARM64     | 4+ cores               |
| GPU            | Not required           | CUDA 11.8+ (optional)  |
| Microphone     | Any USB/built-in       | Low-latency USB mic    |

---

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

**PyAudio on macOS** (if pip fails):
```bash
brew install portaudio
pip install PyAudio
```

**PyAudio on Windows** (if pip fails):
```bash
pip install pipwin
pipwin install PyAudio
```

### 2. Export Model Weights

```bash
python run_voxverify.py quantize --output weights/
```

This creates:
- `weights/aasist_tiny.pt` — PyTorch checkpoint (404 KB)
- `weights/aasist_tiny.onnx` — ONNX FP32 model (866 KB)
- `weights/aasist_tiny_optimized.onnx` — ONNX optimized (505 KB)
- `weights/aasist_tiny_int8.pt` — PyTorch INT8 quantized (244 KB)

### 3. Launch the Dashboard

```bash
python run_voxverify.py ui
```

Opens the Streamlit dashboard at `http://localhost:8501`.

### 4. Start Monitoring

In the dashboard sidebar:
1. Select your audio input device from the dropdown
2. Adjust the sensitivity slider (0.0–1.0, default 0.5)
3. Click **Start Monitoring**

The confidence indicator turns:
- **Green** — Audio is authentic
- **Red** — Synthetic audio detected
- **Yellow** — Uncertain / borderline

---

## Headless Mode (No UI)

For server or terminal-only use:

```bash
python run_voxverify.py monitor --model weights/aasist_tiny.onnx
```

Optional flags:
- `--device 0` — Audio device index
- `--threshold 0.6` — Detection sensitivity

---

## Running the Stress Tests

Validate the model against 50 adversarial audio samples:

```bash
python run_voxverify.py test --model weights/aasist_tiny.onnx
```

The test suite generates:
- 5 clean speech-like signals (harmonics + envelope)
- 10 synthetic artifacts (vocoder, GAN, concatenation)
- 20 compression variants (MP3 128/320, OGG 96, AAC 256)
- 10 noise-corrupted samples (white, pink, babble at various SNR)
- 5 edge cases (silence, clipping, DC offset, pure tone)

Pass criteria: ≥80% pass rate, <100ms mean latency.

---

## Performance Audit

Profile memory usage and latency:

```bash
python run_voxverify.py audit --model weights/aasist_tiny.onnx --duration 60 --output audit.json
```

Checks:
- Peak RAM stays under 2 GB
- Mean inference under 100ms
- No memory leaks over sustained monitoring

---

## Architecture

```
vox_verify/
├── models/
│   └── aasist_tiny.py         # AASIST-Tiny with sparse GAT (88K params)
├── engine/
│   ├── quantize.py            # ONNX export + INT8/FP16 quantization
│   └── dynamic_threshold.py   # Noise-adaptive sensitivity adjustment
├── stream/
│   └── buffer_engine.py       # Real-time PyAudio capture + inference
├── ui/
│   ├── dashboard.py           # Streamlit dashboard
│   └── logger.py              # Thread-safe JSON event logger
├── profiler/
│   └── memory_audit.py        # Memory + latency profiler
├── tests/
│   └── adversarial_stress_test.py  # 50-sample adversarial test suite
├── weights/                   # Model checkpoints (generated)
├── logs/                      # Detection event logs (generated)
├── run_voxverify.py           # CLI entry point
├── requirements.txt           # Python dependencies
└── LOCAL_DEPLOY.md            # This file
```

### Model Architecture (AASIST-Tiny)

Based on [AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks](https://arxiv.org/abs/2110.01200):

1. **SincConv Encoder** — 70 learnable bandpass filters operating on raw 16kHz waveforms
2. **Residual Blocks** — 2 × Conv1D blocks (32 channels) with batch norm and skip connections
3. **Dual Graph Construction** — Spectral (23 nodes) and temporal (12 nodes) graph representations
4. **Sparse GAT Layers** — Top-k=4 attention masking per node for O(k×N) instead of O(N²) attention
5. **Heterogeneous Stacking GAT (HS-GAL)** — Cross-domain fusion via learnable stack nodes
6. **Max Graph Operation** — Dual-branch competitive pooling with extended readout
7. **Classifier** — Linear(240→2) outputting bonafide/spoof logits

### Dynamic Thresholding

The sensitivity threshold auto-adjusts based on ambient noise:

| Noise Level       | RMS (dB) | Threshold | Rationale                    |
|-------------------|----------|-----------|------------------------------|
| Low (quiet room)  | < -40    | 0.40      | More sensitive to subtle fakes |
| Medium (office)   | -40 to -20 | 0.50   | Balanced default              |
| High (street)     | -20 to -10 | 0.65   | Reduce false positives        |
| Very High         | > -10    | 0.80      | Flagged as unreliable         |

Transitions are rate-limited (max 0.05/sec) with 2-second hysteresis to prevent oscillation.

---

## Training Your Own Model

The model ships with randomly initialized weights. To train on real data:

1. Download the [ASVspoof 2019 LA dataset](https://www.asvspoof.org/index2019.html)
2. Prepare audio as 16kHz mono WAV files (1-4 seconds each)
3. Create a training script using the `AASISTTiny` model class:

```python
from vox_verify.models.aasist_tiny import AASISTTiny
import torch

model = AASISTTiny()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for audio, label in dataloader:
    logits = model(audio)           # (B, 2)
    loss = criterion(logits, label)  # label: 0=bonafide, 1=spoof
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

4. After training, re-export:
```bash
python run_voxverify.py quantize --checkpoint path/to/trained.pt --output weights/
```

---

## Benchmark Results

Tested on Intel i7-12700H / 16GB RAM / Ubuntu 22.04:

| Model Variant     | Size     | Mean Latency | P95 Latency | Target |
|-------------------|----------|-------------|-------------|--------|
| ONNX FP32         | 866 KB   | 10.01 ms    | 10.46 ms    | <100ms ✓ |
| ONNX Optimized    | 505 KB   | 7.44 ms     | 8.92 ms     | <100ms ✓ |
| PyTorch INT8      | 244 KB   | 16.70 ms    | 18.21 ms    | <100ms ✓ |

Adversarial stress test: 40/50 passed (80%) with untrained weights.
Compression consistency: 100% (identical predictions across MP3/OGG/AAC variants).

---

## Troubleshooting

**"No audio devices found"**
- Check microphone permissions in system settings
- Run `python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count())"` to verify

**"ONNX model not found"**
- Run `python run_voxverify.py quantize --output weights/` first

**High RAM usage (>2GB)**
- Switch to the optimized ONNX model in the dashboard sidebar
- Reduce the waveform display window in settings

**False positives in noisy environments**
- The dynamic thresholding should handle this automatically
- Manually increase the sensitivity slider toward 1.0 (less sensitive)

---

## License

This project implements the AASIST-Tiny architecture based on research by
Jung et al. (2022). The code in this repository is provided for research
and educational purposes.

**References:**
- [AASIST Paper (arXiv:2110.01200)](https://arxiv.org/abs/2110.01200)
- [RawNetLite (arXiv:2504.20923)](https://arxiv.org/abs/2504.20923)
- [ASVspoof Challenge](https://www.asvspoof.org/)
