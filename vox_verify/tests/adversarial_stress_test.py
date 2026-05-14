"""
adversarial_stress_test.py
==========================

Adversarial stress test suite for the VoxVerify anti-spoofing model.

This module generates 50 deterministic synthetic audio test samples spanning
five categories designed to probe different failure modes of a deepfake-
detection model:

  - **clean**      (5)  – speech-like harmonic signals (bonafide baseline)
  - **artifact**   (10) – vocoder, GAN, and concatenation artifacts
  - **compressed** (20) – 5 base signals × 4 codec quality levels
  - **noisy**      (10) – white noise, pink noise, and babble at varying SNR
  - **edge**       (5)  – silence, DC offset, clipping, pure tone, near-silence

Running the full suite::

    python adversarial_stress_test.py \\
        --model_path /path/to/model.onnx \\
        --output_dir /tmp/adv_tests

Generating samples only (no model required)::

    python adversarial_stress_test.py \\
        --output_dir /tmp/adv_tests \\
        --generate_only

Dependencies
------------
numpy, scipy, onnxruntime (optional), ffmpeg CLI (optional, falls back gracefully)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, lfilter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("vox_verify.adversarial_stress_test")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000          # Hz
GLOBAL_SEED: int = 42              # deterministic everywhere
TOTAL_SAMPLES: int = 50


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class TestResult:
    """Result record for a single test sample.

    Attributes
    ----------
    sample_id : int
        Unique index (0-based) within the 50-sample suite.
    sample_type : str
        Category: ``'clean'``, ``'artifact'``, ``'compressed'``,
        ``'noisy'``, or ``'edge'``.
    compression_type : str
        Codec descriptor (``'none'``, ``'mp3_128'``, ``'mp3_320'``,
        ``'ogg_96'``, ``'aac_256'``) or ``'n/a'`` when not applicable.
    expected_label : str
        Anticipated decision: ``'bonafide'``, ``'spoof'``, or ``'any'``
        (for edge-cases where either label is acceptable).
    predicted_label : str
        Model decision (``'bonafide'`` or ``'spoof'``), or ``'error'``
        when inference failed.
    bonafide_score : float
        Probability / confidence of the bonafide class (0–1).
    spoof_score : float
        Probability / confidence of the spoof class (0–1).
    inference_time_ms : float
        Wall-clock milliseconds spent inside the model forward pass.
    passed : bool
        ``True`` when the predicted label matches the expected label
        (or expected label is ``'any'``).
    error_message : str
        Non-empty if inference raised an exception.
    """

    sample_id: int
    sample_type: str
    compression_type: str
    expected_label: str
    predicted_label: str
    bonafide_score: float
    spoof_score: float
    inference_time_ms: float
    passed: bool
    error_message: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


@dataclass
class TestReport:
    """Aggregated report over all 50 adversarial test samples.

    Attributes
    ----------
    total_tests : int
        Number of samples evaluated.
    passed : int
        Number that matched their expected label (or ``'any'``).
    failed : int
        Number that did not match.
    pass_rate : float
        ``passed / total_tests``.
    by_category : dict
        Per-category breakdown ``{category: {"total", "passed", "failed"}}``.
    compression_consistency_score : float
        Mean fraction of compression variants of the same base signal that
        share the same predicted label.  Range 0–1 (1 = perfectly consistent).
    latency_mean_ms : float
        Mean inference latency across all samples.
    latency_p95_ms : float
        95th-percentile inference latency.
    latency_max_ms : float
        Maximum inference latency.
    failures : list[TestResult]
        Full records for every failed test.
    results : list[TestResult]
        Full records for every test (passed and failed).
    """

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    compression_consistency_score: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    failures: List[TestResult] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of the entire report."""
        d = asdict(self)
        return d

    def to_json(self, path: str | Path) -> None:
        """Write the report as indented JSON to *path*.

        Parameters
        ----------
        path : str or Path
            Destination file.  Parent directories are created if absent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Report saved → %s", path)

    def to_markdown(self) -> str:
        """Render the report as a Markdown string.

        Returns
        -------
        str
            Multi-section Markdown document ready for display or file output.
        """
        lines: List[str] = []

        lines.append("# VoxVerify Adversarial Stress-Test Report\n")

        # --- summary table ---
        lines.append("## Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total tests | {self.total_tests} |")
        lines.append(f"| Passed | {self.passed} |")
        lines.append(f"| Failed | {self.failed} |")
        lines.append(f"| Pass rate | {self.pass_rate:.1%} |")
        lines.append(
            f"| Compression consistency | {self.compression_consistency_score:.3f} |"
        )
        lines.append("")

        # --- latency table ---
        lines.append("## Latency Statistics\n")
        lines.append("| Metric | ms |")
        lines.append("|--------|----|")
        lines.append(f"| Mean | {self.latency_mean_ms:.2f} |")
        lines.append(f"| P95 | {self.latency_p95_ms:.2f} |")
        lines.append(f"| Max | {self.latency_max_ms:.2f} |")
        lines.append("")

        # --- per-category breakdown ---
        lines.append("## Category Breakdown\n")
        lines.append("| Category | Total | Passed | Failed | Pass Rate |")
        lines.append("|----------|-------|--------|--------|-----------|")
        for cat, stats in sorted(self.by_category.items()):
            rate = stats["passed"] / stats["total"] if stats["total"] else 0.0
            lines.append(
                f"| {cat} | {stats['total']} | {stats['passed']} "
                f"| {stats['failed']} | {rate:.1%} |"
            )
        lines.append("")

        # --- failures ---
        if self.failures:
            lines.append("## Failure Analysis\n")
            lines.append(
                "| ID | Type | Codec | Expected | Predicted | "
                "Bonafide | Spoof | Error |"
            )
            lines.append(
                "|----|------|-------|----------|-----------|"
                "---------|-------|-------|"
            )
            for f in self.failures:
                err = f.error_message[:40] if f.error_message else "–"
                lines.append(
                    f"| {f.sample_id} | {f.sample_type} | {f.compression_type} "
                    f"| {f.expected_label} | {f.predicted_label} "
                    f"| {f.bonafide_score:.3f} | {f.spoof_score:.3f} | {err} |"
                )
            lines.append("")
        else:
            lines.append("## Failure Analysis\n\n_No failures._\n")

        return "\n".join(lines)


# ===========================================================================
# Synthetic audio generation
# ===========================================================================

class SyntheticAudioGenerator:
    """Generates 50 deterministic synthetic audio test samples.

    All generation uses **vectorised NumPy** operations (no Python-level
    loops over individual samples).  A fixed random seed guarantees
    reproducibility.

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate in Hz.  Default is 16 000.
    seed : int, optional
        Master seed for all random number generation.  Default is 42.
    output_dir : str or Path, optional
        When supplied, each generated sample is also written to disk as a
        16-bit WAV file in this directory.

    Examples
    --------
    >>> gen = SyntheticAudioGenerator(output_dir="/tmp/samples")
    >>> samples = gen.generate_all()
    >>> len(samples)
    50
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        seed: int = GLOBAL_SEED,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Master RNG – all sub-RNGs derived from this
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> List[Dict]:
        """Generate all 50 test samples and return their metadata.

        Returns
        -------
        list[dict]
            Each entry has keys:
            ``sample_id``, ``sample_type``, ``compression_type``,
            ``expected_label``, ``audio`` (float32 ndarray, –1…1).
        """
        samples: List[Dict] = []

        # Deterministic sub-seeds per category so adding/removing a category
        # does not change samples in other categories.
        samples.extend(self._generate_clean(base_seed=100))          # IDs 0-4
        samples.extend(self._generate_artifacts(base_seed=200))       # IDs 5-14
        samples.extend(self._generate_compressed(base_seed=300))      # IDs 15-34
        samples.extend(self._generate_noisy(base_seed=400))           # IDs 35-44
        samples.extend(self._generate_edge_cases(base_seed=500))      # IDs 45-49

        assert len(samples) == TOTAL_SAMPLES, (
            f"Expected {TOTAL_SAMPLES} samples, got {len(samples)}"
        )

        if self.output_dir:
            for s in samples:
                self._save_wav(s["audio"], s["sample_id"], s["sample_type"])

        return samples

    # ------------------------------------------------------------------
    # Category A – Clean speech-like signals (IDs 0-4)
    # ------------------------------------------------------------------

    def _generate_clean(self, base_seed: int) -> List[Dict]:
        """Generate 5 speech-like harmonic signals with amplitude modulation.

        Each signal combines multiple sine waves at speech-relevant
        fundamental frequencies (100–300 Hz) plus their harmonics up to
        4 kHz, then applies a smoothly varying amplitude envelope that
        mimics the slow amplitude modulation of natural speech.

        Parameters
        ----------
        base_seed : int
            Deterministic seed offset for this category.

        Returns
        -------
        list[dict]
            Five sample descriptors with ``expected_label='bonafide'``.
        """
        rng = np.random.default_rng(base_seed)
        samples = []

        # Five distinct fundamental frequencies in the speech range
        fundamentals = [120.0, 160.0, 200.0, 240.0, 280.0]

        for i, f0 in enumerate(fundamentals):
            duration = float(rng.uniform(1.5, 3.0))
            n = int(duration * self.sample_rate)
            t = np.arange(n, dtype=np.float64) / self.sample_rate

            # Build harmonic stack (up to 4 000 Hz or 12 harmonics)
            audio = np.zeros(n, dtype=np.float64)
            max_harmonic = min(12, int(4000.0 / f0))
            harmonic_amps = 1.0 / np.arange(1, max_harmonic + 1, dtype=np.float64)
            phases = rng.uniform(0, 2 * np.pi, size=max_harmonic)

            # Vectorised harmonic summation
            harmonic_indices = np.arange(1, max_harmonic + 1, dtype=np.float64)
            # shape: (max_harmonic, n)
            freq_matrix = np.outer(harmonic_indices * f0, t)
            phase_matrix = phases[:, np.newaxis]
            audio = (
                harmonic_amps[:, np.newaxis]
                * np.sin(2 * np.pi * freq_matrix + phase_matrix)
            ).sum(axis=0)

            # Amplitude modulation: slow sinusoidal at 3–6 Hz (speech rate)
            am_freq = rng.uniform(3.0, 6.0)
            am_phase = rng.uniform(0, 2 * np.pi)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t + am_phase)
            audio = audio * envelope

            # Normalise to peak ±0.8
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio * (0.8 / peak)

            samples.append(
                self._make_descriptor(
                    sample_id=i,
                    sample_type="clean",
                    compression_type="none",
                    expected_label="bonafide",
                    audio=audio.astype(np.float32),
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Category B – Synthetic artifacts (IDs 5-14)
    # ------------------------------------------------------------------

    def _generate_artifacts(self, base_seed: int) -> List[Dict]:
        """Generate 10 samples containing common deepfake artefacts.

        Three artefact classes are represented:

        * **Vocoder artefacts** (3 samples) – periodic phase discontinuities
          at the vocoder analysis frame rate cause audible buzzing/clicking.
        * **GAN artefacts** (4 samples) – regular spectral holes and narrow
          amplitude peaks in the frequency domain, characteristic of some
          GAN vocoders.
        * **Concatenation artefacts** (3 samples) – abrupt amplitude and
          phase jumps at splice boundaries mimic cut-and-paste forgeries.

        Parameters
        ----------
        base_seed : int
            Deterministic seed offset.

        Returns
        -------
        list[dict]
            Ten sample descriptors with ``expected_label='spoof'``.
        """
        rng = np.random.default_rng(base_seed)
        samples = []

        # --- 3 vocoder samples (IDs 5-7) ---
        for i in range(3):
            duration = float(rng.uniform(1.5, 2.5))
            n = int(duration * self.sample_rate)
            t = np.arange(n, dtype=np.float64) / self.sample_rate

            # Base speech-like signal
            f0 = float(rng.uniform(100.0, 300.0))
            audio = np.sin(2 * np.pi * f0 * t)

            # Phase discontinuity every `frame_shift` samples
            frame_shift = int(0.010 * self.sample_rate)  # 10 ms frames
            frame_boundaries = np.arange(frame_shift, n, frame_shift)
            # Phase jumps are random ±π/4 … ±π
            jump_magnitudes = rng.uniform(np.pi / 4, np.pi, size=len(frame_boundaries))
            # Cumulative phase offset array
            phase_offsets = np.zeros(n, dtype=np.float64)
            for idx, boundary in zip(jump_magnitudes, frame_boundaries):
                phase_offsets[boundary:] += idx * rng.choice([-1.0, 1.0])

            audio = np.sin(2 * np.pi * f0 * t + phase_offsets)
            audio = (audio * 0.7).astype(np.float32)

            samples.append(
                self._make_descriptor(
                    sample_id=5 + i,
                    sample_type="artifact",
                    compression_type="vocoder",
                    expected_label="spoof",
                    audio=audio,
                )
            )

        # --- 4 GAN artefact samples (IDs 8-11) ---
        for i in range(4):
            duration = float(rng.uniform(1.0, 2.0))
            n = int(duration * self.sample_rate)

            # White-noise base, then apply spectral shaping with holes
            audio = rng.standard_normal(n).astype(np.float64)

            # FFT → carve spectral holes + add narrow peaks
            spectrum = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

            # Hole interval: every `hole_spacing` Hz
            hole_spacing = float(rng.choice([200.0, 400.0, 500.0, 250.0]))
            hole_bw = float(rng.uniform(20.0, 60.0))  # bandwidth in Hz
            hole_freqs = np.arange(hole_spacing, 8000.0, hole_spacing)
            for hf in hole_freqs:
                mask = np.abs(freqs - hf) < hole_bw / 2
                spectrum[mask] *= 0.05  # almost silence in these bands

            # Narrow amplitude peaks (aliasing artefacts)
            peak_freqs = rng.uniform(1000.0, 7500.0, size=6)
            for pf in peak_freqs:
                mask = np.abs(freqs - pf) < 15.0
                spectrum[mask] *= 4.0

            audio = np.fft.irfft(spectrum, n=n)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.7

            samples.append(
                self._make_descriptor(
                    sample_id=8 + i,
                    sample_type="artifact",
                    compression_type="gan",
                    expected_label="spoof",
                    audio=audio.astype(np.float32),
                )
            )

        # --- 3 concatenation artefact samples (IDs 12-14) ---
        for i in range(3):
            duration = float(rng.uniform(1.5, 2.5))
            n = int(duration * self.sample_rate)
            t = np.arange(n, dtype=np.float64) / self.sample_rate

            # 2-4 segments with different f0 and phase, joined abruptly
            n_segments = rng.integers(2, 5)
            boundaries = np.sort(rng.integers(n // (n_segments + 1), n, size=n_segments - 1))
            boundaries = np.concatenate([[0], boundaries, [n]])

            audio = np.zeros(n, dtype=np.float64)
            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                seg_len = seg_end - seg_start
                f0_seg = float(rng.uniform(100.0, 400.0))
                phase_seg = rng.uniform(0, 2 * np.pi)
                amp_seg = rng.uniform(0.3, 0.9)
                t_seg = np.arange(seg_len, dtype=np.float64) / self.sample_rate
                audio[seg_start:seg_end] = (
                    amp_seg * np.sin(2 * np.pi * f0_seg * t_seg + phase_seg)
                )

            samples.append(
                self._make_descriptor(
                    sample_id=12 + i,
                    sample_type="artifact",
                    compression_type="concatenation",
                    expected_label="spoof",
                    audio=audio.astype(np.float32),
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Category C – Compression variants (IDs 15-34)
    # ------------------------------------------------------------------

    def _generate_compressed(self, base_seed: int) -> List[Dict]:
        """Generate 5 base signals × 4 codec settings = 20 compressed samples.

        The four codec quality levels are:
        ``mp3_128`` (MP3 128 kbps), ``mp3_320`` (MP3 320 kbps),
        ``ogg_96`` (OGG/Vorbis 96 kbps), ``aac_256`` (AAC 256 kbps).

        When ``ffmpeg`` is available each base signal is piped through the
        codec round-trip (encode → decode → raw PCM).  When ``ffmpeg`` is
        absent the method falls back to pure-NumPy simulation:

        * Low-pass filtering at a codec-specific cutoff frequency.
        * Additive quantisation noise scaled to the codec's nominal bit depth.
        * Spectral band-limiting applied via the FFT.

        Parameters
        ----------
        base_seed : int
            Deterministic seed offset.

        Returns
        -------
        list[dict]
            Twenty sample descriptors with ``expected_label='any'`` because
            compression alone does not change the bonafide/spoof label of the
            underlying signal (but we test prediction *consistency*).
        """
        rng = np.random.default_rng(base_seed)
        samples = []

        # Codec specifications (name, ffmpeg_args, simulated_cutoff_hz, noise_level)
        codecs = [
            ("mp3_128",  ["-codec:a", "libmp3lame", "-b:a", "128k"], 16000, 1e-4),
            ("mp3_320",  ["-codec:a", "libmp3lame", "-b:a", "320k"], 20000, 2e-5),
            ("ogg_96",   ["-codec:a", "libvorbis",  "-b:a", "96k"],  14000, 2e-4),
            ("aac_256",  ["-codec:a", "aac",        "-b:a", "256k"], 20000, 2e-5),
        ]
        has_ffmpeg = _ffmpeg_available()
        if not has_ffmpeg:
            logger.warning(
                "ffmpeg not found – using NumPy-based compression simulation."
            )

        # Generate 5 distinct base signals (same approach as clean, but shorter)
        for base_idx in range(5):
            f0 = float(rng.uniform(120.0, 300.0))
            duration = float(rng.uniform(1.0, 2.0))
            n = int(duration * self.sample_rate)
            t = np.arange(n, dtype=np.float64) / self.sample_rate

            n_harmonics = min(8, int(3500.0 / f0))
            amps = 1.0 / np.arange(1, n_harmonics + 1, dtype=np.float64)
            phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)
            freq_matrix = np.outer(np.arange(1, n_harmonics + 1, dtype=np.float64) * f0, t)
            base_audio = (
                amps[:, np.newaxis] * np.sin(2 * np.pi * freq_matrix + phases[:, np.newaxis])
            ).sum(axis=0)
            base_peak = np.max(np.abs(base_audio))
            if base_peak > 0:
                base_audio = base_audio / base_peak * 0.8

            for codec_name, ffmpeg_args, cutoff, noise_level in codecs:
                sample_id = 15 + base_idx * len(codecs) + codecs.index(
                    (codec_name, ffmpeg_args, cutoff, noise_level)
                )

                if has_ffmpeg:
                    compressed = _compress_with_ffmpeg(
                        base_audio.astype(np.float32),
                        self.sample_rate,
                        ffmpeg_args,
                    )
                    if compressed is None:
                        # ffmpeg failed for this sample – fall back
                        compressed = _simulate_compression(
                            base_audio, self.sample_rate, cutoff, noise_level, rng
                        ).astype(np.float32)
                else:
                    compressed = _simulate_compression(
                        base_audio, self.sample_rate, cutoff, noise_level, rng
                    ).astype(np.float32)

                samples.append(
                    self._make_descriptor(
                        sample_id=sample_id,
                        sample_type="compressed",
                        compression_type=codec_name,
                        expected_label="any",
                        audio=compressed,
                        base_signal_id=base_idx,
                    )
                )

        return samples

    # ------------------------------------------------------------------
    # Category D – Noise-corrupted samples (IDs 35-44)
    # ------------------------------------------------------------------

    def _generate_noisy(self, base_seed: int) -> List[Dict]:
        """Generate 10 noise-corrupted samples.

        Sub-types:

        * **White noise** – 5 SNR levels: 5, 10, 20, 30, 40 dB
          (IDs 35-39, expected ``'any'``).
        * **Pink noise** (1/f spectral shape) at SNR 10 dB (ID 40,
          expected ``'any'``).
        * **Pink noise** at SNR 25 dB (ID 41, expected ``'any'``).
        * **Babble noise** – bandpass-filtered random signals modulated at
          talking rate (IDs 42-44, expected ``'any'``).

        Parameters
        ----------
        base_seed : int
            Deterministic seed offset.

        Returns
        -------
        list[dict]
            Ten sample descriptors.
        """
        rng = np.random.default_rng(base_seed)
        samples = []

        # Shared clean base for all noise variants
        f0 = 180.0
        duration = 2.0
        n = int(duration * self.sample_rate)
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        n_harm = int(4000.0 / f0)
        amps = 1.0 / np.arange(1, n_harm + 1, dtype=np.float64)
        phases = rng.uniform(0, 2 * np.pi, size=n_harm)
        freq_matrix = np.outer(np.arange(1, n_harm + 1, dtype=np.float64) * f0, t)
        clean = (
            amps[:, np.newaxis] * np.sin(2 * np.pi * freq_matrix + phases[:, np.newaxis])
        ).sum(axis=0)
        clean = clean / np.max(np.abs(clean)) * 0.7

        # White noise at 5 SNR levels (IDs 35-39)
        snr_db_levels = [5.0, 10.0, 20.0, 30.0, 40.0]
        for i, snr_db in enumerate(snr_db_levels):
            noise = rng.standard_normal(n)
            audio = _add_noise_at_snr(clean, noise, snr_db)
            samples.append(
                self._make_descriptor(
                    sample_id=35 + i,
                    sample_type="noisy",
                    compression_type=f"white_snr{int(snr_db)}dB",
                    expected_label="any",
                    audio=audio.astype(np.float32),
                )
            )

        # Pink noise at two levels (IDs 40-41)
        for i, snr_db in enumerate([10.0, 25.0]):
            pink = _pink_noise(n, rng)
            audio = _add_noise_at_snr(clean, pink, snr_db)
            samples.append(
                self._make_descriptor(
                    sample_id=40 + i,
                    sample_type="noisy",
                    compression_type=f"pink_snr{int(snr_db)}dB",
                    expected_label="any",
                    audio=audio.astype(np.float32),
                )
            )

        # Babble noise (IDs 42-44)
        for i, snr_db in enumerate([5.0, 15.0, 25.0]):
            babble = _babble_noise(n, self.sample_rate, rng)
            audio = _add_noise_at_snr(clean, babble, snr_db)
            samples.append(
                self._make_descriptor(
                    sample_id=42 + i,
                    sample_type="noisy",
                    compression_type=f"babble_snr{int(snr_db)}dB",
                    expected_label="any",
                    audio=audio.astype(np.float32),
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Category E – Edge cases (IDs 45-49)
    # ------------------------------------------------------------------

    def _generate_edge_cases(self, base_seed: int) -> List[Dict]:
        """Generate 5 boundary / edge-case samples.

        The model must not crash on any of these inputs.

        ===  ========================  ========================  ===========
        ID   Description               Construction              Expected
        ===  ========================  ========================  ===========
        45   Near-silence (−60 dB)     Gaussian noise at −60 dB  ``'any'``
        46   Hard-clipped audio        Clip at ±0.5              ``'any'``
        47   DC offset                 Sine + 0.5 DC             ``'any'``
        48   Pure tone (1 kHz)         Single sine wave          ``'any'``
        49   Complete silence          All-zeros array           ``'any'``
        ===  ========================  ========================  ===========

        Parameters
        ----------
        base_seed : int
            Deterministic seed offset.

        Returns
        -------
        list[dict]
            Five sample descriptors with ``expected_label='any'``.
        """
        rng = np.random.default_rng(base_seed)
        n = int(2.0 * self.sample_rate)  # 2-second edge cases
        t = np.arange(n, dtype=np.float64) / self.sample_rate

        edge_samples = []

        # ID 45 – near-silence (-60 dB ≈ amplitude factor 0.001)
        near_silence = rng.standard_normal(n) * 0.001
        edge_samples.append(("near_silence", near_silence))

        # ID 46 – hard-clipped audio
        base_sine = np.sin(2 * np.pi * 200.0 * t)
        clipped = np.clip(base_sine, -0.5, 0.5)
        edge_samples.append(("clipped", clipped))

        # ID 47 – DC offset
        dc_audio = np.sin(2 * np.pi * 150.0 * t) * 0.5 + 0.5
        edge_samples.append(("dc_offset", dc_audio))

        # ID 48 – pure 1 kHz tone
        pure_tone = np.sin(2 * np.pi * 1000.0 * t) * 0.8
        edge_samples.append(("pure_tone_1kHz", pure_tone))

        # ID 49 – complete silence
        silence = np.zeros(n, dtype=np.float64)
        edge_samples.append(("silence", silence))

        return [
            self._make_descriptor(
                sample_id=45 + idx,
                sample_type="edge",
                compression_type="none",
                expected_label="any",
                audio=audio.astype(np.float32),
                edge_subtype=name,
            )
            for idx, (name, audio) in enumerate(edge_samples)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_descriptor(
        sample_id: int,
        sample_type: str,
        compression_type: str,
        expected_label: str,
        audio: np.ndarray,
        **extra,
    ) -> Dict:
        """Build a sample descriptor dictionary.

        Parameters
        ----------
        sample_id : int
        sample_type : str
        compression_type : str
        expected_label : str
        audio : np.ndarray
            Float32 waveform in [−1, 1].
        **extra
            Additional metadata (e.g. ``base_signal_id``, ``edge_subtype``).
        """
        return {
            "sample_id": sample_id,
            "sample_type": sample_type,
            "compression_type": compression_type,
            "expected_label": expected_label,
            "audio": audio,
            **extra,
        }

    def _save_wav(self, audio: np.ndarray, sample_id: int, sample_type: str) -> None:
        """Write *audio* as a 16-bit WAV file to ``self.output_dir``.

        Parameters
        ----------
        audio : np.ndarray
            Float32 waveform.
        sample_id : int
        sample_type : str
        """
        if self.output_dir is None:
            return
        fname = self.output_dir / f"sample_{sample_id:03d}_{sample_type}.wav"
        # Clip and convert to int16
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        wavfile.write(str(fname), self.sample_rate, audio_int16)


# ===========================================================================
# Module-level audio utility functions
# ===========================================================================

def _ffmpeg_available() -> bool:
    """Return ``True`` if ``ffmpeg`` is on the system PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compress_with_ffmpeg(
    audio: np.ndarray,
    sample_rate: int,
    codec_args: List[str],
) -> Optional[np.ndarray]:
    """Round-trip *audio* through an ffmpeg codec and return the result.

    The pipeline is:

    1. Write float32 raw PCM to a temporary WAV file.
    2. Encode with ffmpeg using *codec_args* to a temporary output file.
    3. Decode back to a WAV file.
    4. Read decoded WAV and return the samples as float32.

    Parameters
    ----------
    audio : np.ndarray
        Input float32 waveform in [−1, 1].
    sample_rate : int
        Sample rate of *audio*.
    codec_args : list[str]
        ffmpeg encoding arguments, e.g. ``['-codec:a', 'libmp3lame', '-b:a', '128k']``.

    Returns
    -------
    np.ndarray or None
        Decoded float32 waveform, or ``None`` if any subprocess call fails.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "input.wav")
            enc_file = os.path.join(tmpdir, "encoded.tmp")
            out_wav = os.path.join(tmpdir, "decoded.wav")

            # Determine appropriate file extension from codec args
            if "libmp3lame" in codec_args:
                enc_file = os.path.join(tmpdir, "encoded.mp3")
            elif "libvorbis" in codec_args:
                enc_file = os.path.join(tmpdir, "encoded.ogg")
            elif "aac" in codec_args:
                enc_file = os.path.join(tmpdir, "encoded.m4a")

            # Step 1: write source WAV
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(in_wav, sample_rate, audio_int16)

            # Step 2: encode
            cmd_encode = (
                ["ffmpeg", "-y", "-loglevel", "error", "-i", in_wav]
                + codec_args
                + [enc_file]
            )
            result = subprocess.run(cmd_encode, capture_output=True, timeout=30)
            if result.returncode != 0:
                logger.debug("ffmpeg encode failed: %s", result.stderr.decode())
                return None

            # Step 3: decode back to WAV
            cmd_decode = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", enc_file,
                "-ar", str(sample_rate),
                "-ac", "1",
                out_wav,
            ]
            result = subprocess.run(cmd_decode, capture_output=True, timeout=30)
            if result.returncode != 0:
                logger.debug("ffmpeg decode failed: %s", result.stderr.decode())
                return None

            # Step 4: read decoded WAV
            sr_out, decoded = wavfile.read(out_wav)
            if decoded.dtype == np.int16:
                decoded = decoded.astype(np.float32) / 32768.0
            elif decoded.dtype == np.int32:
                decoded = decoded.astype(np.float32) / 2**31
            elif decoded.dtype != np.float32:
                decoded = decoded.astype(np.float32)

            # Handle stereo → mono
            if decoded.ndim == 2:
                decoded = decoded.mean(axis=1)

            # Resample length to match input if codec changed it slightly
            if len(decoded) != len(audio):
                # Linear interpolation to match length
                xp = np.linspace(0, 1, len(decoded))
                x = np.linspace(0, 1, len(audio))
                decoded = np.interp(x, xp, decoded).astype(np.float32)

            return decoded

    except Exception as exc:  # noqa: BLE001
        logger.debug("_compress_with_ffmpeg exception: %s", exc)
        return None


def _simulate_compression(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate codec compression artefacts using purely NumPy/SciPy.

    Three effects are combined:

    1. **Low-pass filter** at *cutoff_hz* (Butterworth, order 8).
    2. **Spectral band limiting** via FFT-domain hard clipping.
    3. **Quantisation noise** scaled by *noise_level*.

    Parameters
    ----------
    audio : np.ndarray
        Input float64 waveform.
    sample_rate : int
    cutoff_hz : float
        Low-pass cutoff frequency in Hz.
    noise_level : float
        RMS amplitude of additive quantisation noise.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Processed float64 waveform.
    """
    nyq = sample_rate / 2.0
    normalized_cutoff = min(cutoff_hz / nyq, 0.99)

    sos = butter(8, normalized_cutoff, btype="low", output="sos")
    filtered = sosfilt(sos, audio)

    # Spectral band limiting: zero out components above cutoff_hz
    spectrum = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / sample_rate)
    spectrum[freqs > cutoff_hz] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(audio))

    # Quantisation noise
    q_noise = rng.standard_normal(len(filtered)) * noise_level
    result = filtered + q_noise

    return result.astype(np.float64)


def _add_noise_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Scale *noise* and add it to *signal* at the requested SNR.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal (any float dtype).
    noise : np.ndarray
        Noise array of the same length as *signal*.
    snr_db : float
        Desired signal-to-noise ratio in decibels.

    Returns
    -------
    np.ndarray
        Noisy mixture, same dtype as *signal*, clipped to [−1, 1].
    """
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixture = signal + scale * noise
    return np.clip(mixture, -1.0, 1.0).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f (pink) noise of length *n* via FFT shaping.

    Parameters
    ----------
    n : int
        Number of samples.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Float64 pink noise, zero-mean, unit-variance approximation.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    # 1/f shaping; avoid division by zero at DC
    freqs[0] = 1.0
    pink_spectrum = spectrum / np.sqrt(freqs)
    pink_spectrum[0] = 0.0  # remove DC
    pink = np.fft.irfft(pink_spectrum, n=n)
    # Normalise
    std = np.std(pink)
    if std > 0:
        pink /= std
    return pink


def _babble_noise(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a babble-like noise signal of length *n*.

    Babble is modelled as the sum of several independent bandpass-filtered
    noise streams, each amplitude-modulated at a random talking rate
    (1–6 Hz), simulating overlapping speech from multiple talkers.

    Parameters
    ----------
    n : int
        Number of samples.
    sample_rate : int
        Sample rate in Hz.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Float64 babble noise, normalised to approximately unit variance.
    """
    t = np.arange(n, dtype=np.float64) / sample_rate
    babble = np.zeros(n, dtype=np.float64)
    n_talkers = 6

    for _ in range(n_talkers):
        # Each talker occupies a random 200–1000 Hz band in speech range
        low_hz = float(rng.uniform(100.0, 2000.0))
        high_hz = min(low_hz + float(rng.uniform(200.0, 1000.0)), sample_rate / 2 - 1)
        nyq = sample_rate / 2.0
        low_n = np.clip(low_hz / nyq, 0.01, 0.99)
        high_n = np.clip(high_hz / nyq, 0.01, 0.99)
        if high_n <= low_n:
            high_n = min(low_n + 0.05, 0.99)

        raw = rng.standard_normal(n)
        sos = butter(4, [low_n, high_n], btype="band", output="sos")
        filtered = sosfilt(sos, raw)

        # Amplitude modulation at talker's speaking rate
        talk_rate = float(rng.uniform(1.0, 6.0))
        talk_phase = rng.uniform(0, 2 * np.pi)
        envelope = np.clip(
            np.sin(2 * np.pi * talk_rate * t + talk_phase), 0.0, None
        )
        babble += filtered * envelope

    std = np.std(babble)
    if std > 0:
        babble /= std
    return babble


# ===========================================================================
# Model inference wrapper
# ===========================================================================

class _ModelWrapper:
    """Thin wrapper around ONNX Runtime and PyTorch model inference.

    Attempts to load an ONNX model with ``onnxruntime``.  If the file has a
    ``.pt`` or ``.pth`` extension, falls back to ``torch.jit.load``.
    When *model_path* is ``None`` or loading fails, a **dummy model** that
    returns random scores is used (useful when ``--generate_only`` is set or
    when no trained model is available yet).

    Parameters
    ----------
    model_path : str or Path or None
        Path to the model weights file.
    sample_rate : int
        Expected input sample rate of the model.
    """

    def __init__(
        self,
        model_path: Optional[str | Path],
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        self._session = None   # ONNX Runtime InferenceSession
        self._torch_model = None
        self._use_dummy = False
        self._dummy_rng = np.random.default_rng(GLOBAL_SEED + 99)

        if model_path is None:
            logger.warning("No model path supplied – using dummy inference.")
            self._use_dummy = True
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning("Model file not found: %s – using dummy inference.", model_path)
            self._use_dummy = True
            return

        suffix = model_path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx(model_path)
        elif suffix in {".pt", ".pth"}:
            self._load_torch(model_path)
        else:
            logger.warning(
                "Unrecognised model extension '%s' – using dummy inference.", suffix
            )
            self._use_dummy = True

    def _load_onnx(self, path: Path) -> None:
        """Load an ONNX model with onnxruntime."""
        try:
            import onnxruntime as ort  # noqa: PLC0415
            self._session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("Loaded ONNX model: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load ONNX model (%s) – using dummy.", exc)
            self._use_dummy = True

    def _load_torch(self, path: Path) -> None:
        """Load a TorchScript model with torch.jit."""
        try:
            import torch  # noqa: PLC0415
            self._torch_model = torch.jit.load(str(path), map_location="cpu")
            self._torch_model.eval()
            logger.info("Loaded TorchScript model: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load Torch model (%s) – using dummy.", exc)
            self._use_dummy = True

    def infer(self, audio: np.ndarray) -> Tuple[float, float]:
        """Run inference on *audio* and return (bonafide_score, spoof_score).

        Parameters
        ----------
        audio : np.ndarray
            Float32 1-D waveform.

        Returns
        -------
        tuple[float, float]
            ``(bonafide_score, spoof_score)`` in [0, 1] summing to 1.

        Raises
        ------
        RuntimeError
            If model inference raises an unexpected exception (callers should
            catch this and record it in the test result).
        """
        if self._use_dummy:
            return self._dummy_infer(audio)

        if self._session is not None:
            return self._onnx_infer(audio)

        if self._torch_model is not None:
            return self._torch_infer(audio)

        return self._dummy_infer(audio)

    @staticmethod
    def _fix_length(audio: np.ndarray, target: int = 16_000) -> np.ndarray:
        """Pad (zero) or truncate *audio* to exactly *target* samples."""
        if audio.shape[-1] >= target:
            return audio[:target]
        padded = np.zeros(target, dtype=audio.dtype)
        padded[: audio.shape[-1]] = audio
        return padded

    def _onnx_infer(self, audio: np.ndarray) -> Tuple[float, float]:
        """Run inference through the ONNX Runtime session."""
        audio = self._fix_length(audio, 16_000)
        # Typical anti-spoofing models expect (batch=1, channels=1, samples)
        x = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: x})
        # Assume first output is logits or probabilities of shape (1, 2)
        raw = np.array(outputs[0]).flatten()
        return _to_probabilities(raw)

    def _torch_infer(self, audio: np.ndarray) -> Tuple[float, float]:
        """Run inference through a TorchScript model."""
        import torch  # noqa: PLC0415
        audio = self._fix_length(audio, 16_000)
        x = torch.from_numpy(audio[np.newaxis, np.newaxis, :].astype(np.float32))
        with torch.no_grad():
            raw = self._torch_model(x).numpy().flatten()
        return _to_probabilities(raw)

    def _dummy_infer(self, audio: np.ndarray) -> Tuple[float, float]:
        """Produce random-but-reproducible scores (used when no model is loaded)."""
        # Use signal energy as a weak heuristic so edge cases differ slightly
        energy = float(np.mean(audio ** 2))
        # Near-silence → lean bonafide; high energy → more uncertain
        base_bonafide = 0.5 + 0.3 * np.exp(-10.0 * energy)
        noise = float(self._dummy_rng.uniform(-0.1, 0.1))
        bonafide = float(np.clip(base_bonafide + noise, 0.01, 0.99))
        return bonafide, 1.0 - bonafide


def _to_probabilities(raw: np.ndarray) -> Tuple[float, float]:
    """Convert raw model output to (bonafide_prob, spoof_prob).

    Handles both logit-space (2-class softmax) and probability outputs.
    If the output has only one element it is interpreted as the bonafide
    probability directly.

    Parameters
    ----------
    raw : np.ndarray
        Flattened model output.

    Returns
    -------
    tuple[float, float]
    """
    if len(raw) == 1:
        bonafide = float(np.clip(raw[0], 0.0, 1.0))
        return bonafide, 1.0 - bonafide

    if len(raw) >= 2:
        # Apply softmax to first two logits
        logits = raw[:2].astype(np.float64)
        logits -= logits.max()  # numerical stability
        exps = np.exp(logits)
        probs = exps / exps.sum()
        return float(probs[0]), float(probs[1])

    return 0.5, 0.5


# ===========================================================================
# Adversarial test runner
# ===========================================================================

class AdversarialTestRunner:
    """Runs the 50-sample adversarial suite against a VoxVerify model.

    Parameters
    ----------
    model_path : str or Path or None
        Path to an ONNX or TorchScript model file.  Pass ``None`` to run in
        dummy mode (useful for sample generation and pipeline testing).
    output_dir : str or Path or None
        If given, generated WAV files are saved here and the report JSON is
        written to ``<output_dir>/report.json``.
    sample_rate : int, optional
        Audio sample rate expected by the model.  Default 16 000.
    seed : int, optional
        Master random seed for reproducibility.  Default 42.

    Examples
    --------
    >>> runner = AdversarialTestRunner(model_path="model.onnx", output_dir="/tmp/out")
    >>> report = runner.run_all()
    >>> print(report.to_markdown())
    """

    # Thresholds for binary decision
    BONAFIDE_THRESHOLD: float = 0.5

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        sample_rate: int = SAMPLE_RATE,
        seed: int = GLOBAL_SEED,
    ) -> None:
        self.sample_rate = sample_rate
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None

        self._generator = SyntheticAudioGenerator(
            sample_rate=sample_rate,
            seed=seed,
            output_dir=output_dir,
        )
        self._model = _ModelWrapper(model_path, sample_rate)
        self._samples: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_samples(self) -> List[Dict]:
        """Generate (or regenerate) all 50 test samples.

        Returns
        -------
        list[dict]
            Sample descriptors including the ``'audio'`` array.
        """
        logger.info("Generating %d synthetic test samples…", TOTAL_SAMPLES)
        self._samples = self._generator.generate_all()
        logger.info("Sample generation complete.")
        return self._samples

    def run_all(self) -> TestReport:
        """Run all 50 adversarial tests and return a full ``TestReport``.

        If samples have not been generated yet, :meth:`load_samples` is
        called automatically.

        Returns
        -------
        TestReport
            Aggregated results, statistics, and failure analysis.
        """
        if self._samples is None:
            self.load_samples()

        logger.info("Running inference on %d samples…", len(self._samples))
        results: List[TestResult] = []
        for sample in self._samples:
            result = self.run_single(sample["sample_id"])
            results.append(result)

        report = self._build_report(results)

        if self.output_dir:
            report_path = Path(self.output_dir) / "report.json"
            report.to_json(report_path)
            md_path = Path(self.output_dir) / "report.md"
            md_path.write_text(report.to_markdown(), encoding="utf-8")
            logger.info("Markdown report saved → %s", md_path)

        return report

    def run_single(self, sample_id: int) -> TestResult:
        """Run inference on a single sample identified by *sample_id*.

        Parameters
        ----------
        sample_id : int
            Index in [0, 49] referring to one of the 50 test samples.

        Returns
        -------
        TestResult
            Full result record for this sample.

        Notes
        -----
        If :meth:`load_samples` has not been called, it is called here.
        If *sample_id* is not found, a failing ``TestResult`` with
        ``error_message`` is returned instead of raising.
        """
        if self._samples is None:
            self.load_samples()

        # Find the sample
        sample = next(
            (s for s in self._samples if s["sample_id"] == sample_id), None
        )
        if sample is None:
            return TestResult(
                sample_id=sample_id,
                sample_type="unknown",
                compression_type="unknown",
                expected_label="any",
                predicted_label="error",
                bonafide_score=0.0,
                spoof_score=0.0,
                inference_time_ms=0.0,
                passed=False,
                error_message=f"Sample ID {sample_id} not found in generated set.",
            )

        audio: np.ndarray = sample["audio"]
        expected: str = sample["expected_label"]

        # Run inference with timing
        t0 = time.perf_counter()
        try:
            bonafide_score, spoof_score = self._model.infer(audio)
            predicted_label = (
                "bonafide" if bonafide_score >= self.BONAFIDE_THRESHOLD else "spoof"
            )
            error_message = ""
        except Exception as exc:  # noqa: BLE001
            bonafide_score = 0.0
            spoof_score = 0.0
            predicted_label = "error"
            error_message = str(exc)
            logger.error(
                "Inference error on sample %d (%s): %s",
                sample_id,
                sample["sample_type"],
                exc,
            )

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        # Evaluate pass condition
        passed = _evaluate_pass(
            expected_label=expected,
            predicted_label=predicted_label,
            sample_type=sample["sample_type"],
        )

        result = TestResult(
            sample_id=sample_id,
            sample_type=sample["sample_type"],
            compression_type=sample["compression_type"],
            expected_label=expected,
            predicted_label=predicted_label,
            bonafide_score=bonafide_score,
            spoof_score=spoof_score,
            inference_time_ms=inference_time_ms,
            passed=passed,
            error_message=error_message,
        )

        logger.debug(
            "Sample %02d [%s/%s] → %s (bonafide=%.3f, spoof=%.3f, %.1f ms) %s",
            sample_id,
            sample["sample_type"],
            sample["compression_type"],
            predicted_label,
            bonafide_score,
            spoof_score,
            inference_time_ms,
            "✓" if passed else "✗",
        )

        return result

    # ------------------------------------------------------------------
    # Internal report builder
    # ------------------------------------------------------------------

    def _build_report(self, results: List[TestResult]) -> TestReport:
        """Aggregate a list of ``TestResult`` objects into a ``TestReport``.

        Parameters
        ----------
        results : list[TestResult]

        Returns
        -------
        TestReport
        """
        total = len(results)
        passed_count = sum(r.passed for r in results)
        failed_count = total - passed_count

        # Per-category stats
        categories = sorted({r.sample_type for r in results})
        by_category: Dict[str, Dict[str, int]] = {}
        for cat in categories:
            cat_results = [r for r in results if r.sample_type == cat]
            by_category[cat] = {
                "total": len(cat_results),
                "passed": sum(r.passed for r in cat_results),
                "failed": sum(not r.passed for r in cat_results),
            }

        # Compression consistency
        consistency = _compression_consistency(results)

        # Latency statistics (only include successful inferences)
        latencies = np.array(
            [r.inference_time_ms for r in results if r.predicted_label != "error"],
            dtype=np.float64,
        )
        if len(latencies) > 0:
            latency_mean = float(np.mean(latencies))
            latency_p95 = float(np.percentile(latencies, 95))
            latency_max = float(np.max(latencies))
        else:
            latency_mean = latency_p95 = latency_max = 0.0

        failures = [r for r in results if not r.passed]

        return TestReport(
            total_tests=total,
            passed=passed_count,
            failed=failed_count,
            pass_rate=passed_count / total if total > 0 else 0.0,
            by_category=by_category,
            compression_consistency_score=consistency,
            latency_mean_ms=latency_mean,
            latency_p95_ms=latency_p95,
            latency_max_ms=latency_max,
            failures=failures,
            results=results,
        )


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def _evaluate_pass(
    expected_label: str,
    predicted_label: str,
    sample_type: str,
) -> bool:
    """Determine whether a single test passes.

    Rules
    -----
    * If *expected_label* is ``'any'`` and *predicted_label* is not
      ``'error'``, the test passes (edge cases and noisy samples must not
      crash the model, but any classification is acceptable).
    * If *expected_label* is ``'bonafide'`` or ``'spoof'``, the test passes
      only when *predicted_label* matches exactly.
    * Prediction of ``'error'`` always fails.

    Parameters
    ----------
    expected_label : str
    predicted_label : str
    sample_type : str
        Unused currently; reserved for future category-specific logic.

    Returns
    -------
    bool
    """
    if predicted_label == "error":
        return False
    if expected_label == "any":
        return True
    return predicted_label == expected_label


def _compression_consistency(results: List[TestResult]) -> float:
    """Compute the compression consistency score.

    For each group of compressed variants derived from the same base signal
    (identified by shared ``sample_type='compressed'`` and consecutive ID
    block), count the fraction that share the **majority** predicted label.
    The score is the mean fraction across all five base-signal groups.

    A score of 1.0 means the model always predicts the same label for all
    four codec variants of the same base signal.

    Parameters
    ----------
    results : list[TestResult]

    Returns
    -------
    float
        Consistency score in [0, 1].
    """
    compressed = [r for r in results if r.sample_type == "compressed"]
    if not compressed:
        return 1.0

    # Group into blocks of 4 (mp3_128, mp3_320, ogg_96, aac_256)
    n_codecs = 4
    n_bases = len(compressed) // n_codecs
    if n_bases == 0:
        return 1.0

    group_consistencies: List[float] = []
    for b in range(n_bases):
        group = compressed[b * n_codecs : (b + 1) * n_codecs]
        labels = [r.predicted_label for r in group if r.predicted_label != "error"]
        if not labels:
            group_consistencies.append(0.0)
            continue
        unique, counts = np.unique(labels, return_counts=True)
        majority_count = int(counts.max())
        group_consistencies.append(majority_count / len(labels))

    return float(np.mean(group_consistencies))


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "VoxVerify Adversarial Stress Test Suite – "
            "generates 50 synthetic audio samples and evaluates model robustness."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to ONNX (.onnx) or TorchScript (.pt/.pth) model file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated WAV samples and the test report.",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate audio samples; skip model inference.",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help=(
            "Skip sample generation and re-use existing samples in output_dir. "
            "Requires --output_dir to point to an existing sample directory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=SAMPLE_RATE,
        help="Audio sample rate in Hz.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failures detected, 2 = error).
    """
    args = _parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Validate flag combinations
    # ------------------------------------------------------------------
    if args.generate_only and args.test_only:
        logger.error("--generate_only and --test_only are mutually exclusive.")
        return 2

    if args.test_only and not args.output_dir:
        logger.error("--test_only requires --output_dir pointing to existing samples.")
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else None

    # ------------------------------------------------------------------
    # Generate-only mode
    # ------------------------------------------------------------------
    if args.generate_only:
        gen = SyntheticAudioGenerator(
            sample_rate=args.sample_rate,
            seed=args.seed,
            output_dir=output_dir,
        )
        samples = gen.generate_all()
        print(f"Generated {len(samples)} samples.", end="")
        if output_dir:
            print(f"  Saved to: {output_dir}")
        else:
            print()
        return 0

    # ------------------------------------------------------------------
    # Normal or test-only mode
    # ------------------------------------------------------------------
    runner = AdversarialTestRunner(
        model_path=args.model_path,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    if args.test_only:
        # Load samples from existing WAVs in output_dir
        logger.info(
            "--test_only: loading existing WAV files from %s", output_dir
        )
        samples = _load_samples_from_dir(output_dir, args.sample_rate)
        if not samples:
            logger.error("No WAV files found in %s.", output_dir)
            return 2
        runner._samples = samples  # noqa: SLF001
    else:
        runner.load_samples()

    report = runner.run_all()

    # Print summary to stdout
    print()
    print(report.to_markdown())
    print()

    # Save report
    if output_dir:
        report_path = output_dir / "report.json"
        report.to_json(report_path)
        print(f"Full report saved → {report_path}")

    # Exit code
    return 0 if report.failed == 0 else 1


def _load_samples_from_dir(
    directory: Path,
    sample_rate: int,
) -> List[Dict]:
    """Load existing WAV files from *directory* for --test_only mode.

    File names must follow the pattern ``sample_NNN_TYPE.wav`` as written
    by :meth:`SyntheticAudioGenerator._save_wav`.

    Parameters
    ----------
    directory : Path
    sample_rate : int

    Returns
    -------
    list[dict]
        Sample descriptors (without ``'audio'`` arrays initially loaded from
        the original generator; the audio is read from the WAV files).
    """
    import re  # noqa: PLC0415

    samples: List[Dict] = []
    pattern = re.compile(r"sample_(\d{3})_(\w+)\.wav", re.IGNORECASE)

    for wav_path in sorted(directory.glob("sample_*.wav")):
        m = pattern.match(wav_path.name)
        if not m:
            continue
        sample_id = int(m.group(1))
        sample_type = m.group(2)

        sr, data = wavfile.read(str(wav_path))
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2**31
        else:
            audio = data.astype(np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        samples.append(
            {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "compression_type": "loaded_from_disk",
                "expected_label": "any",
                "audio": audio,
            }
        )

    return samples


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
