"""
vox_verify.stream.buffer_engine
================================
Real-time audio stream processor for near real-time deepfake detection.

Architecture
------------
StreamCapture  →  AudioBuffer  →  InferenceWorker  →  callback(DetectionResult)
    |                  |                 |
  PyAudio          circular           ONNX RT
  callback         numpy arr          softmax

Key design goals:
- Zero-copy audio ingestion via np.frombuffer
- Pre-allocated arrays in hot path (no per-frame heap allocations)
- Thread-safe handoff between capture and inference threads
- Vectorised NumPy preprocessing throughout (no Python for-loops)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ─── Optional heavy dependencies (graceful degradation) ──────────────────────
try:
    import pyaudio  # type: ignore
    _PYAUDIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    pyaudio = None  # type: ignore
    _PYAUDIO_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_AVAILABLE = False

# ─── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Structured result emitted after each inference pass.

    Attributes
    ----------
    timestamp:          Unix epoch (seconds) of the frame's leading edge.
    bonafide_score:     Probability [0, 1] that the audio is genuine speech.
    spoof_score:        Probability [0, 1] that the audio is synthesised / replayed.
    is_spoof:           True when ``spoof_score`` exceeds the configured threshold.
    latency_ms:         Wall-clock latency from frame capture to inference completion.
    audio_rms:          RMS energy of the raw frame (useful for SNR gating).
    """
    timestamp: float
    bonafide_score: float
    spoof_score: float
    is_spoof: bool
    latency_ms: float
    audio_rms: float


@dataclass
class EngineConfig:
    """Configuration bag for :class:`BufferEngine`.

    Attributes
    ----------
    sample_rate:            Audio sample rate in Hz (default 16 000).
    buffer_size:            Circular buffer capacity in samples (default 16 000 = 1 s).
    hop_size:               Advance between successive frames (default 8 000 = 0.5 s).
    chunk_size:             PyAudio callback chunk in frames (default 1 024).
    channels:               Number of input channels (default 1 – mono).
    device_index:           PyAudio device index; ``None`` → system default.
    model_path:             Path to the ONNX model file.
    sensitivity_threshold:  ``spoof_score`` cut-off for ``is_spoof`` flag (default 0.5).
    """
    sample_rate: int = 16_000
    buffer_size: int = 16_000
    hop_size: int = 8_000
    chunk_size: int = 1_024
    channels: int = 1
    device_index: Optional[int] = None
    model_path: str = "model.onnx"
    sensitivity_threshold: float = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# AudioBuffer
# ──────────────────────────────────────────────────────────────────────────────

class AudioBuffer:
    """Thread-safe circular audio buffer backed by a pre-allocated NumPy array.

    The buffer holds ``buffer_size`` float32 samples.  Audio is written in
    variable-length chunks via :meth:`push`.  Once at least ``buffer_size``
    samples have accumulated, :meth:`get_frame` returns a contiguous snapshot
    and advances the internal read pointer by ``hop_size`` samples (overlapping
    window semantics).

    Parameters
    ----------
    buffer_size:
        Total capacity in samples.  Equals the inference frame length (default
        16 000, i.e. 1 second at 16 kHz).
    hop_size:
        How far the read pointer advances each time a frame is consumed
        (default 8 000 → 50 % overlap).

    Thread safety
    -------------
    A single :class:`threading.Lock` serialises all reads and writes so that
    one PyAudio callback thread and one inference thread can operate concurrently
    without data races.
    """

    def __init__(
        self,
        buffer_size: int = 16_000,
        hop_size: int = 8_000,
    ) -> None:
        if hop_size > buffer_size:
            raise ValueError("hop_size must be ≤ buffer_size")

        self.buffer_size = buffer_size
        self.hop_size = hop_size

        # Pre-allocate the ring storage and the output snapshot once.
        self._ring: np.ndarray = np.zeros(buffer_size * 2, dtype=np.float32)
        self._frame_buf: np.ndarray = np.zeros(buffer_size, dtype=np.float32)

        self._write_pos: int = 0   # absolute sample counter (mod 2*buffer_size)
        self._fill: int = 0        # how many valid samples are present
        self._lock: threading.Lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray) -> None:
        """Append *chunk* (float32) to the ring.

        The chunk may be any length; the buffer wraps automatically.
        This is called from the PyAudio callback thread.

        Parameters
        ----------
        chunk:
            1-D float32 array of audio samples.  Must already be normalised to
            [-1, 1] by the caller **or** raw int16 converted via
            :func:`numpy.frombuffer` + scaling – either is fine, normalisation
            happens in :class:`InferenceWorker`.
        """
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        n = len(chunk)
        cap = len(self._ring)

        with self._lock:
            end = self._write_pos + n
            if end <= cap:
                self._ring[self._write_pos:end] = chunk
            else:
                # Wrap around: split into two slices.
                first = cap - self._write_pos
                self._ring[self._write_pos:] = chunk[:first]
                self._ring[: n - first] = chunk[first:]

            self._write_pos = end % cap
            self._fill = min(self._fill + n, cap)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the oldest ``buffer_size`` samples, then advance
        the logical read pointer by ``hop_size``.

        Returns ``None`` when fewer than ``buffer_size`` samples are available.

        The returned array is a **copy** stored in a pre-allocated buffer so the
        caller owns it without risk of it being overwritten.
        """
        with self._lock:
            if self._fill < self.buffer_size:
                return None

            cap = len(self._ring)
            # Compute where the oldest sample starts (read_pos = write_pos - fill).
            read_pos = (self._write_pos - self._fill) % cap
            end = read_pos + self.buffer_size

            if end <= cap:
                np.copyto(self._frame_buf, self._ring[read_pos:end])
            else:
                first = cap - read_pos
                self._frame_buf[:first] = self._ring[read_pos:]
                self._frame_buf[first:] = self._ring[: self.buffer_size - first]

            # Advance the read pointer by consuming hop_size samples.
            self._fill -= self.hop_size

            return self._frame_buf.copy()

    def is_ready(self) -> bool:
        """Return ``True`` when a full frame is available for retrieval."""
        with self._lock:
            return self._fill >= self.buffer_size

    def clear(self) -> None:
        """Reset the buffer to an empty state (zero-fills backing array)."""
        with self._lock:
            self._ring[:] = 0.0
            self._frame_buf[:] = 0.0
            self._write_pos = 0
            self._fill = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AudioBuffer(buffer_size={self.buffer_size}, "
            f"hop_size={self.hop_size}, fill={self._fill})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# StreamCapture
# ──────────────────────────────────────────────────────────────────────────────

class StreamCapture:
    """Non-blocking PyAudio microphone / loopback capture.

    Audio is delivered to a user-supplied *data_callback* in the PyAudio
    callback thread without copying (zero-copy via :func:`numpy.frombuffer`).
    Each delivery is a 1-D float32 array scaled to [-1, 1].

    Parameters
    ----------
    sample_rate:    Capture sample rate (default 16 000 Hz).
    channels:       Number of audio channels (default 1 = mono).
    chunk_size:     Frames per PyAudio callback (default 1 024).
    device_index:   PyAudio device index; ``None`` → system default.
    data_callback:  Called with each chunk as a float32 NumPy array.

    Raises
    ------
    RuntimeError:
        If PyAudio is not installed or no input device is available.
    PermissionError:
        If the OS denies microphone access.
    """

    _INT16_MAX: float = 32768.0  # scaling constant for int16 → float32 normalisation

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_size: int = 1_024,
        device_index: Optional[int] = None,
        data_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        if not _PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio is not installed.  "
                "Install it with: pip install pyaudio"
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.data_callback = data_callback

        self._pa: Optional["pyaudio.PyAudio"] = None
        self._stream: Optional["pyaudio.Stream"] = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()

        # Pre-allocate conversion buffer to avoid per-callback heap allocations.
        self._conv_buf: np.ndarray = np.zeros(chunk_size * channels, dtype=np.float32)

    # ── device discovery ──────────────────────────────────────────────────────

    def list_devices(self) -> List[dict]:
        """Return a list of available audio devices with metadata.

        Each entry is a dict with keys:
        ``index``, ``name``, ``maxInputChannels``, ``defaultSampleRate``,
        ``hostApi``, ``isDefaultInput``.

        Returns
        -------
        list[dict]
            All devices that have at least one input channel.
        """
        pa = pyaudio.PyAudio()
        devices: List[dict] = []
        try:
            default_idx: Optional[int] = None
            try:
                info = pa.get_default_input_device_info()
                default_idx = int(info["index"])
            except OSError:
                pass

            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    if int(info.get("maxInputChannels", 0)) > 0:
                        devices.append(
                            {
                                "index": i,
                                "name": info.get("name", f"Device {i}"),
                                "maxInputChannels": int(info.get("maxInputChannels", 0)),
                                "defaultSampleRate": float(
                                    info.get("defaultSampleRate", 0.0)
                                ),
                                "hostApi": int(info.get("hostApi", 0)),
                                "isDefaultInput": i == default_idx,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not query device %d: %s", i, exc)
        finally:
            pa.terminate()

        logger.info("Found %d input device(s).", len(devices))
        return devices

    def set_device(self, index: int) -> None:
        """Switch to a different input device.

        If a stream is currently active it will be restarted on the new device.

        Parameters
        ----------
        index:
            PyAudio device index as returned by :meth:`list_devices`.
        """
        was_running = self._running
        if was_running:
            self.stop()
        self.device_index = index
        logger.info("Audio input device set to index %d.", index)
        if was_running:
            self.start()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the PyAudio stream and begin capturing.

        Raises
        ------
        RuntimeError:
            If already running or no suitable input device can be found.
        PermissionError:
            If the OS rejects microphone access.
        """
        with self._lock:
            if self._running:
                logger.warning("StreamCapture.start() called while already running.")
                return

            self._pa = pyaudio.PyAudio()

            # Validate / select device.
            if self.device_index is not None:
                try:
                    info = self._pa.get_device_info_by_index(self.device_index)
                    if int(info.get("maxInputChannels", 0)) < 1:
                        raise RuntimeError(
                            f"Device {self.device_index} ({info.get('name')}) "
                            "has no input channels."
                        )
                except OSError as exc:
                    self._pa.terminate()
                    raise RuntimeError(
                        f"Device index {self.device_index} is invalid: {exc}"
                    ) from exc
            else:
                try:
                    self._pa.get_default_input_device_info()
                except OSError as exc:
                    self._pa.terminate()
                    raise RuntimeError(
                        "No default input device found.  "
                        "Connect a microphone or select a device via set_device()."
                    ) from exc

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._pyaudio_callback,
                )
            except OSError as exc:
                self._pa.terminate()
                if "Permission" in str(exc) or "denied" in str(exc).lower():
                    raise PermissionError(
                        "Microphone access denied by the operating system.  "
                        "Check your system privacy settings."
                    ) from exc
                raise RuntimeError(f"Could not open audio stream: {exc}") from exc

            self._stream.start_stream()
            self._running = True
            logger.info(
                "StreamCapture started (device=%s, rate=%d, chunk=%d).",
                self.device_index,
                self.sample_rate,
                self.chunk_size,
            )

    def stop(self) -> None:
        """Stop capture and release all PyAudio resources."""
        with self._lock:
            if not self._running:
                return
            self._running = False

            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error closing PyAudio stream: %s", exc)
                finally:
                    self._stream = None

            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error terminating PyAudio: %s", exc)
                finally:
                    self._pa = None

            logger.info("StreamCapture stopped.")

    # ── internal PyAudio callback ─────────────────────────────────────────────

    def _pyaudio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status_flags: int,
    ) -> tuple:
        """PyAudio non-blocking stream callback.

        Converts raw int16 bytes to float32 samples using vectorised NumPy ops
        (zero Python loops) and forwards to the user-supplied *data_callback*.
        """
        if not self._running:
            return (None, pyaudio.paComplete)

        if status_flags:
            logger.debug("PyAudio status flags: %d (possible under/overflow).", status_flags)

        # Zero-copy view of the PCM bytes as int16, then vectorised cast + scale.
        pcm_int16: np.ndarray = np.frombuffer(in_data, dtype=np.int16)

        # Vectorised normalisation: int16 → float32 in [-1, 1].
        # np.multiply with out= reuses the pre-allocated buffer when sizes match.
        n = pcm_int16.size
        if n == len(self._conv_buf):
            np.multiply(pcm_int16, 1.0 / self._INT16_MAX, out=self._conv_buf)
            audio_f32 = self._conv_buf
        else:
            # Chunk size mismatch (e.g. final fragment) – allocate once.
            audio_f32 = pcm_int16.astype(np.float32) * (1.0 / self._INT16_MAX)

        # For multi-channel audio, mix down to mono via vectorised mean.
        if self.channels > 1:
            audio_f32 = audio_f32.reshape(-1, self.channels).mean(axis=1)

        if self.data_callback is not None:
            try:
                self.data_callback(audio_f32)
            except Exception as exc:  # noqa: BLE001
                logger.error("data_callback raised an exception: %s", exc, exc_info=True)

        return (None, pyaudio.paContinue)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StreamCapture(rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"device={self.device_index}, "
            f"running={self._running})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NoiseEstimator
# ──────────────────────────────────────────────────────────────────────────────

class NoiseEstimator:
    """Running background-noise level estimator using exponential moving average.

    Uses the RMS energy of each audio chunk as a proxy for signal level.  The
    noise floor is tracked with an EMA so short loud events do not inflate the
    estimate permanently.

    The estimate is deliberately conservative: the EMA alpha is small so the
    noise level only rises slowly and falls quickly, which tends to produce
    better SNR gate behaviour in practice.

    Parameters
    ----------
    alpha:
        EMA smoothing factor in (0, 1).  Smaller values → slower adaptation.
        Default 0.05 (≈ 20-sample time constant).
    initial_noise:
        Starting estimate of the noise floor (default 1e-4 ≈ −80 dBFS).

    Notes
    -----
    This class is used by Phase 8 (dynamic thresholding) to decide when a
    frame is too quiet to warrant inference.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        initial_noise: float = 1e-4,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")
        self._alpha = alpha
        self._noise_level: float = initial_noise
        self._lock: threading.Lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def current_noise_level(self) -> float:
        """Current EMA estimate of the background noise RMS energy."""
        with self._lock:
            return self._noise_level

    def update(self, frame: np.ndarray) -> float:
        """Feed a new audio frame and return the updated noise estimate.

        Parameters
        ----------
        frame:
            1-D float32 audio array.

        Returns
        -------
        float
            Updated noise-floor estimate after processing *frame*.
        """
        # Vectorised RMS – single NumPy expression, no Python loop.
        rms: float = float(np.sqrt(np.mean(np.square(frame))))
        with self._lock:
            # EMA update.
            self._noise_level = (
                self._alpha * rms + (1.0 - self._alpha) * self._noise_level
            )
            return self._noise_level

    def signal_to_noise_ratio(self, frame: np.ndarray) -> float:
        """Compute the instantaneous SNR of *frame* relative to the noise floor.

        Returns
        -------
        float
            SNR as a linear ratio (not dB).  Values < 1.0 indicate frame is
            below the current noise floor estimate.
        """
        rms: float = float(np.sqrt(np.mean(np.square(frame))))
        noise = self.current_noise_level
        if noise < 1e-12:
            return float("inf")
        return rms / noise

    def reset(self, initial_noise: float = 1e-4) -> None:
        """Reset the noise estimate to *initial_noise*."""
        with self._lock:
            self._noise_level = initial_noise

    def __repr__(self) -> str:  # pragma: no cover
        return f"NoiseEstimator(alpha={self._alpha}, noise_level={self._noise_level:.6f})"


# ──────────────────────────────────────────────────────────────────────────────
# InferenceWorker
# ──────────────────────────────────────────────────────────────────────────────

class InferenceWorker:
    """Dedicated inference thread that drains :class:`AudioBuffer` frames.

    Frames are preprocessed (amplitude normalisation, float32 tensor) and
    passed to an ONNX Runtime session.  Softmax converts raw logits to
    probabilities; results are forwarded via *result_callback*.

    Parameters
    ----------
    buffer:
        The :class:`AudioBuffer` to pull frames from.
    model_path:
        File-system path to the ``.onnx`` model.
    result_callback:
        Called in the worker thread with each :class:`DetectionResult`.
    sensitivity_threshold:
        ``spoof_score`` threshold above which ``is_spoof`` is set to ``True``
        (default 0.5).
    poll_interval:
        Seconds between buffer readiness checks (default 0.01 = 10 ms).

    Raises
    ------
    RuntimeError:
        If ONNX Runtime is not installed.
    """

    def __init__(
        self,
        buffer: AudioBuffer,
        model_path: str,
        result_callback: Callable[[DetectionResult], None],
        sensitivity_threshold: float = 0.5,
        poll_interval: float = 0.01,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "ONNX Runtime is not installed.  "
                "Install it with: pip install onnxruntime"
            )

        self._buffer = buffer
        self._model_path = model_path
        self._result_callback = result_callback
        self._threshold = sensitivity_threshold
        self._poll_interval = poll_interval

        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

        # Pre-allocate softmax work buffer (2 logits: bonafide, spoof).
        self._softmax_buf: np.ndarray = np.zeros(2, dtype=np.float64)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load the ONNX model and start the inference thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("InferenceWorker is already running.")
            return

        logger.info("Loading ONNX model from '%s' …", self._model_path)
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = 2
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            self._session = ort.InferenceSession(
                self._model_path, sess_options=sess_opts
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info(
                "Model loaded. Input name: '%s'.", self._input_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX model '{self._model_path}': {exc}"
            ) from exc

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="inference-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("InferenceWorker thread started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker thread to stop and wait for it to join.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for the thread to finish (default 5 s).
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "InferenceWorker thread did not stop within %.1f s.", timeout
                )
        self._session = None
        logger.info("InferenceWorker stopped.")

    # ── worker loop ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: poll the buffer and run inference when a frame is ready."""
        logger.debug("InferenceWorker._run() entered.")

        while not self._stop_event.is_set():
            if not self._buffer.is_ready():
                self._stop_event.wait(timeout=self._poll_interval)
                continue

            frame = self._buffer.get_frame()
            if frame is None:
                continue

            capture_time = time.monotonic()
            frame_timestamp = time.time()

            try:
                result = self._process_frame(frame, frame_timestamp, capture_time)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Inference error: %s", exc, exc_info=True
                )
                continue

            try:
                self._result_callback(result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "result_callback raised an exception: %s", exc, exc_info=True
                )

        logger.debug("InferenceWorker._run() exiting.")

    # ── preprocessing + inference ─────────────────────────────────────────────

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        capture_time: float,
    ) -> DetectionResult:
        """Preprocess *frame*, run ONNX inference, and return a :class:`DetectionResult`.

        Preprocessing pipeline (all vectorised NumPy):
        1. Compute RMS energy (used for SNR gating / reporting).
        2. Peak-normalise to [-1, 1] (avoid clipping artefacts for the model).
        3. Reshape to model input shape ``[1, N]``.
        4. Run ONNX session.
        5. Apply softmax to raw logits.
        """
        # ── Step 1: RMS ───────────────────────────────────────────────────────
        # Vectorised: np.sqrt(mean(x²)) — single expression, no loops.
        audio_rms: float = float(np.sqrt(np.mean(np.square(frame))))

        # ── Step 2: peak normalisation ────────────────────────────────────────
        peak: float = float(np.abs(frame).max())
        if peak > 1e-8:
            # In-place multiply to avoid extra allocation.
            normalised = frame * (1.0 / peak)
        else:
            normalised = frame  # silence frame: pass through as-is

        # ── Step 3: shape to [batch=1, samples] ──────────────────────────────
        tensor: np.ndarray = normalised.reshape(1, -1).astype(np.float32)

        # ── Step 4: ONNX inference ────────────────────────────────────────────
        assert self._session is not None, "ONNX session is not initialised."
        raw_output = self._session.run(None, {self._input_name: tensor})
        logits: np.ndarray = np.asarray(raw_output[0], dtype=np.float64).ravel()

        # ── Step 5: softmax (stable version, vectorised) ──────────────────────
        shifted = logits - logits.max()          # numerical stability
        np.exp(shifted, out=self._softmax_buf[:len(shifted)])
        self._softmax_buf[:len(shifted)] /= self._softmax_buf[:len(shifted)].sum()

        bonafide_score: float = float(self._softmax_buf[0])
        spoof_score: float = float(self._softmax_buf[1])
        is_spoof: bool = spoof_score >= self._threshold

        latency_ms: float = (time.monotonic() - capture_time) * 1_000.0

        return DetectionResult(
            timestamp=timestamp,
            bonafide_score=bonafide_score,
            spoof_score=spoof_score,
            is_spoof=is_spoof,
            latency_ms=latency_ms,
            audio_rms=audio_rms,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"InferenceWorker(model='{self._model_path}', "
            f"threshold={self._threshold}, "
            f"running={self._thread is not None and self._thread.is_alive()})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BufferEngine  (main orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class BufferEngine:
    """Top-level orchestrator that wires together capture, buffering, and inference.

    Lifecycle::

        engine = BufferEngine(config)
        engine.on_detection = my_handler   # register callbacks
        engine.on_error     = err_handler
        engine.start()
        …
        engine.stop()

    All resources (PyAudio streams, ONNX session, threads) are released on
    :meth:`stop`.  The engine may be restarted after stopping.

    Parameters
    ----------
    config:
        :class:`EngineConfig` instance with all tunable parameters.

    Event callbacks
    ---------------
    on_detection : ``Callable[[DetectionResult], None]``
        Invoked in the inference thread for every processed frame.
    on_error : ``Callable[[Exception], None]``
        Invoked when an unrecoverable error occurs in any sub-component.
    on_device_change : ``Callable[[int], None]``
        Invoked when the active device index changes.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config: EngineConfig = config or EngineConfig()

        # Public event hooks (set before calling start()).
        self.on_detection: Optional[Callable[[DetectionResult], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_device_change: Optional[Callable[[int], None]] = None

        # Sub-components — created lazily in start().
        self._buffer: Optional[AudioBuffer] = None
        self._capture: Optional[StreamCapture] = None
        self._worker: Optional[InferenceWorker] = None
        self._noise_estimator: Optional[NoiseEstimator] = None

        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise all sub-components and begin the capture→inference pipeline.

        Raises
        ------
        RuntimeError:
            If already running or if required dependencies are missing.
        PermissionError:
            If microphone access is denied.
        """
        with self._lock:
            if self._running:
                logger.warning("BufferEngine.start() called while already running.")
                return

            logger.info("BufferEngine starting …")

            cfg = self.config

            # 1. Audio buffer.
            self._buffer = AudioBuffer(
                buffer_size=cfg.buffer_size,
                hop_size=cfg.hop_size,
            )

            # 2. Noise estimator.
            self._noise_estimator = NoiseEstimator()

            # 3. Stream capture (wired to buffer + noise estimator).
            if not _PYAUDIO_AVAILABLE:
                raise RuntimeError(
                    "PyAudio is required for StreamCapture.  "
                    "Install it with: pip install pyaudio"
                )
            self._capture = StreamCapture(
                sample_rate=cfg.sample_rate,
                channels=cfg.channels,
                chunk_size=cfg.chunk_size,
                device_index=cfg.device_index,
                data_callback=self._on_audio_chunk,
            )

            # 4. Inference worker.
            if not _ORT_AVAILABLE:
                raise RuntimeError(
                    "ONNX Runtime is required for InferenceWorker.  "
                    "Install it with: pip install onnxruntime"
                )
            self._worker = InferenceWorker(
                buffer=self._buffer,
                model_path=cfg.model_path,
                result_callback=self._on_detection_result,
                sensitivity_threshold=cfg.sensitivity_threshold,
            )

            # Start in dependency order: worker first (model loaded), then capture.
            try:
                self._worker.start()
            except Exception as exc:
                logger.error("Failed to start InferenceWorker: %s", exc)
                self._emit_error(exc)
                raise

            try:
                self._capture.start()
            except Exception as exc:
                logger.error("Failed to start StreamCapture: %s", exc)
                self._worker.stop()
                self._emit_error(exc)
                raise

            self._running = True
            logger.info("BufferEngine is running.")

    def stop(self) -> None:
        """Gracefully stop capture and inference, releasing all resources."""
        with self._lock:
            if not self._running:
                return

            logger.info("BufferEngine stopping …")
            self._running = False

            # Stop in reverse dependency order.
            if self._capture is not None:
                try:
                    self._capture.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error stopping StreamCapture: %s", exc)
                finally:
                    self._capture = None

            if self._worker is not None:
                try:
                    self._worker.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error stopping InferenceWorker: %s", exc)
                finally:
                    self._worker = None

            if self._buffer is not None:
                self._buffer.clear()
                self._buffer = None

            self._noise_estimator = None
            logger.info("BufferEngine stopped.")

    def list_devices(self) -> List[dict]:
        """Delegate to :class:`StreamCapture` to enumerate input devices.

        Returns
        -------
        list[dict]
            See :meth:`StreamCapture.list_devices` for the dict schema.
        """
        if not _PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available; cannot list audio devices.")
            return []
        # Instantiate a temporary capture handle purely for device enumeration.
        temp = StreamCapture(sample_rate=self.config.sample_rate)
        return temp.list_devices()

    def set_device(self, index: int) -> None:
        """Hot-swap the input device (restarts capture if currently running).

        Parameters
        ----------
        index:
            PyAudio device index as returned by :meth:`list_devices`.
        """
        self.config.device_index = index
        if self._capture is not None:
            try:
                self._capture.set_device(index)
            except Exception as exc:
                logger.error("Failed to switch device to %d: %s", index, exc)
                self._emit_error(exc)
                raise

        if self.on_device_change is not None:
            try:
                self.on_device_change(index)
            except Exception as exc:  # noqa: BLE001
                logger.debug("on_device_change callback raised: %s", exc)

    @property
    def is_running(self) -> bool:
        """``True`` while the engine is actively capturing and inferring."""
        return self._running

    @property
    def noise_estimator(self) -> Optional[NoiseEstimator]:
        """Expose the active :class:`NoiseEstimator` for Phase 8 dynamic thresholding."""
        return self._noise_estimator

    # ── internal callbacks (hot path) ─────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Receive a normalised float32 chunk from :class:`StreamCapture`.

        Called in the PyAudio callback thread.  Must not block.
        """
        if self._buffer is not None:
            self._buffer.push(chunk)
        if self._noise_estimator is not None:
            self._noise_estimator.update(chunk)

    def _on_detection_result(self, result: DetectionResult) -> None:
        """Receive a :class:`DetectionResult` from :class:`InferenceWorker`.

        Called in the inference thread.
        """
        logger.debug(
            "Detection: spoof=%.3f bonafide=%.3f is_spoof=%s latency=%.1fms rms=%.5f",
            result.spoof_score,
            result.bonafide_score,
            result.is_spoof,
            result.latency_ms,
            result.audio_rms,
        )
        if self.on_detection is not None:
            try:
                self.on_detection(result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "on_detection callback raised an exception: %s",
                    exc,
                    exc_info=True,
                )

    def _emit_error(self, exc: Exception) -> None:
        """Forward *exc* to the ``on_error`` callback if registered."""
        if self.on_error is not None:
            try:
                self.on_error(exc)
            except Exception as cb_exc:  # noqa: BLE001
                logger.debug("on_error callback itself raised: %s", cb_exc)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BufferEngine(running={self._running}, "
            f"model='{self.config.model_path}', "
            f"device={self.config.device_index})"
        )

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "BufferEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.stop()
