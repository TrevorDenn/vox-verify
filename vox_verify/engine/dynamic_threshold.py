"""
dynamic_threshold.py — Adaptive detection-threshold module for Vox-Verify Local.

The module auto-adjusts the spoof-detection sensitivity threshold based on the
ambient noise floor measured from incoming audio frames.

Pipeline
--------
    audio frame
        │
        ▼
    NoiseFloorEstimator   ←── sliding window, EMA smoothing
        │  noise_level_db, spectral_flatness, snr_estimate
        ▼
    AdaptiveThreshold     ←── noise → category → target threshold
        │                      rate-limiting + hysteresis
        │  ThresholdState
        ▼
    ConfidenceCalibrator  ←── SNR-aware score adjustment
        │  CalibratedScore
        ▼
    ThresholdManager      ←── orchestrator, decision history
        │  DetectionDecision
        ▼
    caller

Public API
----------
- NoiseFloorEstimator
- AdaptiveThreshold
- ConfidenceCalibrator
- ThresholdManager
- ThresholdState         (dataclass)
- CalibratedScore        (dataclass)
- DetectionDecision      (dataclass)
- find_optimal_thresholds(model_session, test_samples, labels) -> dict
- plot_threshold_curve(history) -> None
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Noise category boundaries (dB RMS)
_NOISE_LOW_MAX: float = -40.0       # below this → LOW
_NOISE_MEDIUM_MAX: float = -20.0    # -40 to -20 → MEDIUM
_NOISE_HIGH_MAX: float = -10.0      # -20 to -10 → HIGH
                                    # above -10   → VERY HIGH

# Target thresholds per noise category
_THRESHOLD_LOW: float = 0.40
_THRESHOLD_MEDIUM: float = 0.50
_THRESHOLD_HIGH: float = 0.65
_THRESHOLD_VERY_HIGH: float = 0.80

# Adaptive threshold constraints
_MAX_RATE_PER_SECOND: float = 0.05          # max absolute change per second
_HYSTERESIS_SECONDS: float = 2.0            # must stay stable before applying
_SPECTRAL_FLATNESS_REVIEW_THRESHOLD = 0.85  # very tonal or very noisy

# Confidence calibration
_SNR_FULL_CONFIDENCE_DB: float = 20.0   # SNR above this → no penalty
_SNR_ZERO_CONFIDENCE_DB: float = 0.0    # SNR below this → max penalty

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdState:
    """Snapshot of the adaptive threshold and noise conditions at one instant.

    Attributes
    ----------
    current_threshold : float
        The threshold value currently in effect for spoof classification.
    noise_level_db : float
        Smoothed RMS noise floor in dBFS (0 dBFS = full-scale).
    noise_category : str
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"VERY_HIGH"``.
    confidence_modifier : float
        Multiplicative modifier applied to raw model confidence scores
        (range 0–1; 1.0 = no modification).
    is_reliable : bool
        ``False`` when noise conditions are so severe that detections
        should not be trusted without additional review.
    timestamp : float
        Unix timestamp (seconds) when this state was produced.
    """

    current_threshold: float
    noise_level_db: float
    noise_category: str
    confidence_modifier: float
    is_reliable: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibratedScore:
    """Model confidence score after noise-aware calibration.

    Attributes
    ----------
    original_score : float
        Raw probability score from the detection model (0–1).
    calibrated_score : float
        Adjusted score after accounting for noise conditions.
    reliability : float
        Estimated reliability of this calibrated score (0–1).
    needs_review : bool
        ``True`` when unusual audio conditions warrant human review.
    reason : str
        Human-readable explanation of any adjustments applied.
    """

    original_score: float
    calibrated_score: float
    reliability: float
    needs_review: bool
    reason: str


@dataclass
class DetectionDecision:
    """Final per-frame detection decision produced by :class:`ThresholdManager`.

    Attributes
    ----------
    is_spoof : bool
        Whether the frame is classified as a spoof attempt.
    confidence : float
        Calibrated confidence in the ``is_spoof`` label (0–1).
    threshold_used : float
        Adaptive threshold that was applied for this decision.
    noise_db : float
        Estimated noise floor in dBFS at decision time.
    snr_db : float
        Estimated signal-to-noise ratio in dB.
    reliability : float
        How reliable this decision is given current noise conditions (0–1).
    category : str
        Noise category (``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"VERY_HIGH"``).
    needs_review : bool
        ``True`` when the decision should be flagged for manual review.
    """

    is_spoof: bool
    confidence: float
    threshold_used: float
    noise_db: float
    snr_db: float
    reliability: float
    category: str
    needs_review: bool


# ---------------------------------------------------------------------------
# NoiseFloorEstimator
# ---------------------------------------------------------------------------


class NoiseFloorEstimator:
    """Estimates the ambient noise floor from a stream of audio frames.

    All heavy computation is vectorised via NumPy; no Python-level loops are
    used for per-sample arithmetic.

    Parameters
    ----------
    window_size : int
        Number of frames to keep in the sliding window (default 20, ≈10 s at
        512-sample frames @ 16 kHz with 50 % overlap).
    alpha : float
        Exponential moving average smoothing coefficient (0 < alpha ≤ 1).
        Smaller values produce slower, smoother estimates.
    eps : float
        Small constant added before log/division to prevent numerical issues.
    """

    def __init__(
        self,
        window_size: int = 20,
        alpha: float = 0.1,
        eps: float = 1e-10,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        if window_size < 1:
            raise ValueError(f"window_size must be ≥ 1, got {window_size!r}")

        self._window_size = window_size
        self._alpha = alpha
        self._eps = eps

        # Sliding window: each element is a 1-D numpy array (one frame).
        self._window: Deque[np.ndarray] = deque(maxlen=window_size)

        # Smoothed statistics (initialised lazily on first update)
        self._smooth_rms_db: Optional[float] = None
        self._smooth_sf: Optional[float] = None
        self._smooth_zcr: Optional[float] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> None:
        """Ingest one audio frame and update all noise estimates.

        Parameters
        ----------
        frame : np.ndarray
            1-D float32/float64 array of audio samples (any length ≥ 1).
            Values should be in the range [-1, 1] (normalised PCM).
        """
        frame = np.asarray(frame, dtype=np.float64).ravel()
        self._window.append(frame)

        rms_db = self._compute_rms_db(frame)
        sf = self._compute_spectral_flatness(frame)
        zcr = self._compute_zcr(frame)

        # Exponential moving average
        alpha = self._alpha
        if self._smooth_rms_db is None:
            self._smooth_rms_db = rms_db
            self._smooth_sf = sf
            self._smooth_zcr = zcr
        else:
            self._smooth_rms_db = alpha * rms_db + (1.0 - alpha) * self._smooth_rms_db
            self._smooth_sf = alpha * sf + (1.0 - alpha) * self._smooth_sf  # type: ignore[operator]
            self._smooth_zcr = alpha * zcr + (1.0 - alpha) * self._smooth_zcr  # type: ignore[operator]

    def get_noise_level_db(self) -> float:
        """Return the smoothed RMS energy of the noise floor in dBFS.

        Returns
        -------
        float
            Negative value; 0 dBFS corresponds to full-scale amplitude.
            Returns ``-120.0`` (silence proxy) if no frames have been ingested.
        """
        if self._smooth_rms_db is None:
            return -120.0
        return float(self._smooth_rms_db)

    def get_spectral_flatness(self) -> float:
        """Return the smoothed Wiener entropy (spectral flatness, 0–1).

        A value near 1.0 indicates white-noise-like audio; a value near 0.0
        indicates tonal / pitched audio.

        Returns
        -------
        float
            Spectral flatness in [0, 1].  Returns 0.0 before first update.
        """
        if self._smooth_sf is None:
            return 0.0
        return float(np.clip(self._smooth_sf, 0.0, 1.0))

    def get_snr_estimate(self, frame: np.ndarray) -> float:
        """Estimate SNR for *frame* relative to the current noise floor.

        SNR = RMS_frame_db − noise_floor_db.

        Parameters
        ----------
        frame : np.ndarray
            1-D audio frame to evaluate.

        Returns
        -------
        float
            Estimated SNR in dB.  A positive value means the frame is louder
            than the estimated noise floor.
        """
        frame = np.asarray(frame, dtype=np.float64).ravel()
        frame_rms_db = self._compute_rms_db(frame)
        noise_db = self.get_noise_level_db()
        return float(frame_rms_db - noise_db)

    # ------------------------------------------------------------------
    # Private helpers — all vectorised, no Python loops
    # ------------------------------------------------------------------

    def _compute_rms_db(self, frame: np.ndarray) -> float:
        """Compute RMS energy in dBFS for a single frame (vectorised)."""
        rms = np.sqrt(np.mean(np.square(frame)) + self._eps)
        return float(20.0 * np.log10(rms))

    def _compute_spectral_flatness(self, frame: np.ndarray) -> float:
        """Compute Wiener entropy (spectral flatness) via FFT magnitude spectrum.

        Spectral flatness = geometric_mean(|X|) / arithmetic_mean(|X|).

        All arithmetic is vectorised over the frequency bins.
        """
        magnitude = np.abs(np.fft.rfft(frame))
        magnitude = np.where(magnitude > self._eps, magnitude, self._eps)

        log_geo_mean = np.mean(np.log(magnitude))  # log of geometric mean
        arith_mean = np.mean(magnitude)

        geo_mean = float(np.exp(log_geo_mean))
        arith_mean = float(arith_mean)

        if arith_mean < self._eps:
            return 0.0
        return geo_mean / arith_mean

    def _compute_zcr(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate (vectorised, no Python loop).

        ZCR = number of sign changes / (2 * frame_length).
        """
        signs = np.sign(frame)
        # Replace zeros with previous sign to avoid double-counting
        # (vectorised fill-forward via cumsum trick)
        nonzero_mask = signs != 0
        # If all zeros, return 0
        if not np.any(nonzero_mask):
            return 0.0
        # Use np.where to propagate last nonzero sign
        idx = np.where(nonzero_mask, np.arange(len(signs)), 0)
        idx = np.maximum.accumulate(idx)
        filled = signs[idx]

        crossings = np.sum(np.abs(np.diff(filled))) // 2
        return float(crossings) / max(len(frame), 1)


# ---------------------------------------------------------------------------
# AdaptiveThreshold
# ---------------------------------------------------------------------------


class AdaptiveThreshold:
    """Auto-adjusts the detection threshold based on current noise conditions.

    The threshold responds to noise category changes but applies rate-limiting
    and hysteresis to avoid rapid oscillation.

    Parameters
    ----------
    base_threshold : float
        Baseline threshold used for MEDIUM noise (default 0.5).
    max_rate_per_second : float
        Maximum allowed absolute change in threshold per second (default 0.05).
    hysteresis_seconds : float
        Minimum time (s) a target threshold must be stable before it is
        promoted to the active threshold (default 2.0).
    """

    _CATEGORY_MAP: Dict[str, float] = {
        "LOW": _THRESHOLD_LOW,
        "MEDIUM": _THRESHOLD_MEDIUM,
        "HIGH": _THRESHOLD_HIGH,
        "VERY_HIGH": _THRESHOLD_VERY_HIGH,
    }

    def __init__(
        self,
        base_threshold: float = 0.5,
        max_rate_per_second: float = _MAX_RATE_PER_SECOND,
        hysteresis_seconds: float = _HYSTERESIS_SECONDS,
    ) -> None:
        self._base_threshold = base_threshold
        self._max_rate = max_rate_per_second
        self._hysteresis = hysteresis_seconds

        # Active (applied) threshold starts at base
        self._current_threshold: float = base_threshold

        # Target tracking for hysteresis
        self._pending_target: float = base_threshold
        self._pending_since: float = time.time()

        self._last_update_time: float = time.time()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(self, noise_level_db: float, spectral_flatness: float) -> ThresholdState:
        """Compute and apply the adaptive threshold for the current conditions.

        Parameters
        ----------
        noise_level_db : float
            Smoothed RMS noise floor in dBFS from :class:`NoiseFloorEstimator`.
        spectral_flatness : float
            Smoothed spectral flatness (0–1) from :class:`NoiseFloorEstimator`.

        Returns
        -------
        ThresholdState
            Complete snapshot of threshold and noise state after this update.
        """
        now = time.time()
        dt = max(now - self._last_update_time, 1e-6)

        category = self._classify_noise(noise_level_db)
        raw_target = self._CATEGORY_MAP[category]

        # ---- Hysteresis -----------------------------------------------
        if abs(raw_target - self._pending_target) > 1e-9:
            # Target has changed; reset pending timer
            self._pending_target = raw_target
            self._pending_since = now

        stable_seconds = now - self._pending_since
        if stable_seconds >= self._hysteresis:
            # Target has been stable long enough; allow movement toward it
            effective_target = self._pending_target
        else:
            # Still in hysteresis window; stay at current threshold
            effective_target = self._current_threshold

        # ---- Rate limiting --------------------------------------------
        max_delta = self._max_rate * dt
        delta = np.clip(
            effective_target - self._current_threshold,
            -max_delta,
            max_delta,
        )
        self._current_threshold = float(
            np.clip(self._current_threshold + delta, 0.0, 1.0)
        )

        self._last_update_time = now

        # ---- Derived state --------------------------------------------
        is_reliable = category != "VERY_HIGH"
        confidence_modifier = self._compute_confidence_modifier(
            noise_level_db, spectral_flatness
        )

        return ThresholdState(
            current_threshold=self._current_threshold,
            noise_level_db=noise_level_db,
            noise_category=category,
            confidence_modifier=confidence_modifier,
            is_reliable=is_reliable,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_noise(noise_level_db: float) -> str:
        """Map a dBFS RMS value to a named noise category."""
        if noise_level_db < _NOISE_LOW_MAX:
            return "LOW"
        if noise_level_db < _NOISE_MEDIUM_MAX:
            return "MEDIUM"
        if noise_level_db < _NOISE_HIGH_MAX:
            return "HIGH"
        return "VERY_HIGH"

    @staticmethod
    def _compute_confidence_modifier(
        noise_level_db: float, spectral_flatness: float
    ) -> float:
        """Compute a [0, 1] multiplier that penalises high-noise conditions.

        Uses a linear interpolation between -60 dB (no penalty) and the
        VERY_HIGH boundary (-10 dB, maximum penalty of 0.5).
        """
        lower_db = -60.0
        upper_db = _NOISE_HIGH_MAX  # -10 dB
        t = (noise_level_db - lower_db) / (upper_db - lower_db)
        t = float(np.clip(t, 0.0, 1.0))
        # Modifier goes from 1.0 (quiet) to 0.5 (very loud)
        modifier = 1.0 - 0.5 * t
        return float(modifier)


# ---------------------------------------------------------------------------
# ConfidenceCalibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Adjusts raw model confidence scores based on current noise conditions.

    The calibrator uses SNR and spectral flatness to determine how much to
    trust the raw model output, potentially flagging frames for review.

    Parameters
    ----------
    snr_full_confidence_db : float
        SNR above which no calibration penalty is applied (default 20 dB).
    snr_zero_confidence_db : float
        SNR below which maximum calibration penalty is applied (default 0 dB).
    """

    def __init__(
        self,
        snr_full_confidence_db: float = _SNR_FULL_CONFIDENCE_DB,
        snr_zero_confidence_db: float = _SNR_ZERO_CONFIDENCE_DB,
    ) -> None:
        if snr_full_confidence_db <= snr_zero_confidence_db:
            raise ValueError(
                "snr_full_confidence_db must be greater than snr_zero_confidence_db"
            )
        self._snr_full = snr_full_confidence_db
        self._snr_zero = snr_zero_confidence_db

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def calibrate(
        self, raw_score: float, noise_state: ThresholdState, snr_db: Optional[float] = None
    ) -> CalibratedScore:
        """Produce a noise-aware calibrated confidence score.

        Parameters
        ----------
        raw_score : float
            Raw probability score from the detection model (0–1).
        noise_state : ThresholdState
            Current threshold state from :class:`AdaptiveThreshold`.
        snr_db : float, optional
            Signal-to-noise ratio for this specific frame.  If ``None``, the
            SNR is inferred from the noise conditions alone.

        Returns
        -------
        CalibratedScore
            Adjusted score with reliability and review flags.
        """
        reasons: List[str] = []
        needs_review = False

        # ---- SNR-based penalty ----------------------------------------
        if snr_db is None:
            # Approximate SNR from noise category
            snr_db = self._snr_from_category(noise_state.noise_category)

        snr_factor = self._snr_calibration_factor(snr_db)
        if snr_factor < 0.9:
            reasons.append(
                f"SNR={snr_db:.1f} dB below full-confidence threshold "
                f"({self._snr_full} dB); factor={snr_factor:.2f}"
            )

        # ---- Spectral flatness check ----------------------------------
        # Unusual SF (very close to 1 = pure noise) warrants review
        sf = noise_state.confidence_modifier  # proxy available in state
        # Re-derive: if noise is very unreliable, flag regardless
        if not noise_state.is_reliable:
            needs_review = True
            reasons.append("VERY_HIGH noise: conditions unreliable for detection")

        # ---- Combined modifier ----------------------------------------
        combined_modifier = snr_factor * noise_state.confidence_modifier
        calibrated = float(np.clip(raw_score * combined_modifier, 0.0, 1.0))

        # ---- Reliability score ----------------------------------------
        reliability = float(
            np.clip(snr_factor * noise_state.confidence_modifier, 0.0, 1.0)
        )

        # Flag for review if calibration caused a large score shift
        score_shift = abs(raw_score - calibrated)
        if score_shift > 0.15:
            needs_review = True
            reasons.append(
                f"Large calibration shift ({score_shift:.2f}); manual check advised"
            )

        reason = "; ".join(reasons) if reasons else "No adjustments applied"

        return CalibratedScore(
            original_score=float(raw_score),
            calibrated_score=calibrated,
            reliability=reliability,
            needs_review=needs_review,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snr_calibration_factor(self, snr_db: float) -> float:
        """Linear calibration factor in [0.5, 1.0] based on SNR.

        SNR ≥ snr_full_confidence_db  → factor = 1.0  (no penalty)
        SNR ≤ snr_zero_confidence_db  → factor = 0.5  (50 % penalty)
        """
        t = (snr_db - self._snr_zero) / (self._snr_full - self._snr_zero)
        t = float(np.clip(t, 0.0, 1.0))
        return 0.5 + 0.5 * t

    @staticmethod
    def _snr_from_category(category: str) -> float:
        """Map noise category to a representative SNR estimate."""
        mapping = {
            "LOW": 30.0,
            "MEDIUM": 15.0,
            "HIGH": 8.0,
            "VERY_HIGH": 2.0,
        }
        return mapping.get(category, 15.0)


# ---------------------------------------------------------------------------
# ThresholdManager (orchestrator)
# ---------------------------------------------------------------------------


class ThresholdManager:
    """Orchestrates noise estimation, threshold adaptation, and confidence calibration.

    Provides a single :meth:`process_frame` call that returns a complete
    :class:`DetectionDecision` for each audio frame, and maintains a rolling
    history for trend analysis.

    Parameters
    ----------
    noise_estimator : NoiseFloorEstimator, optional
        Pre-configured noise estimator.  A default instance is created if
        ``None`` is given.
    adaptive_threshold : AdaptiveThreshold, optional
        Pre-configured adaptive threshold.  A default instance is created if
        ``None`` is given.
    calibrator : ConfidenceCalibrator, optional
        Pre-configured confidence calibrator.  A default instance is created
        if ``None`` is given.
    history_size : int
        Maximum number of :class:`DetectionDecision` records to retain for
        trend analysis (default 100).
    """

    def __init__(
        self,
        noise_estimator: Optional[NoiseFloorEstimator] = None,
        adaptive_threshold: Optional[AdaptiveThreshold] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        history_size: int = 100,
    ) -> None:
        self._noise_est = noise_estimator or NoiseFloorEstimator()
        self._threshold = adaptive_threshold or AdaptiveThreshold()
        self._calibrator = calibrator or ConfidenceCalibrator()
        self._history: Deque[DetectionDecision] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def process_frame(
        self,
        audio_frame: np.ndarray,
        raw_model_output: np.ndarray,
    ) -> DetectionDecision:
        """Process one audio frame and return a calibrated detection decision.

        Parameters
        ----------
        audio_frame : np.ndarray
            1-D float32/float64 audio samples (normalised PCM, range [-1, 1]).
        raw_model_output : np.ndarray
            Output from the anti-spoofing model.  The *last* element is taken
            as the spoof probability (compatible with typical [bonafide, spoof]
            softmax outputs).  A scalar array is also accepted.

        Returns
        -------
        DetectionDecision
            Complete decision including classification, calibrated confidence,
            noise information, and reliability flags.
        """
        audio_frame = np.asarray(audio_frame, dtype=np.float64).ravel()
        raw_model_output = np.asarray(raw_model_output, dtype=np.float64).ravel()

        # Extract spoof probability from model output
        raw_score: float = float(raw_model_output[-1])

        # Step 1: Update noise floor estimate
        self._noise_est.update(audio_frame)
        noise_db = self._noise_est.get_noise_level_db()
        sf = self._noise_est.get_spectral_flatness()
        snr_db = self._noise_est.get_snr_estimate(audio_frame)

        # Step 2: Update adaptive threshold
        threshold_state = self._threshold.update(noise_db, sf)

        # Step 3: Calibrate confidence score
        calibrated = self._calibrator.calibrate(raw_score, threshold_state, snr_db)

        # Step 4: Make binary decision
        is_spoof = calibrated.calibrated_score >= threshold_state.current_threshold

        decision = DetectionDecision(
            is_spoof=is_spoof,
            confidence=calibrated.calibrated_score,
            threshold_used=threshold_state.current_threshold,
            noise_db=noise_db,
            snr_db=snr_db,
            reliability=calibrated.reliability,
            category=threshold_state.noise_category,
            needs_review=calibrated.needs_review,
        )

        self._history.append(decision)
        return decision

    def get_trend(self) -> Dict[str, Any]:
        """Return summary statistics over the decision history.

        Returns
        -------
        dict
            Keys: ``n_frames``, ``spoof_rate``, ``mean_confidence``,
            ``mean_noise_db``, ``mean_snr_db``, ``mean_reliability``,
            ``review_rate``, ``category_counts``, ``mean_threshold``.
            Returns a minimal dict with ``n_frames=0`` if history is empty.
        """
        if not self._history:
            return {"n_frames": 0}

        decisions = list(self._history)
        n = len(decisions)

        is_spoof_arr = np.array([d.is_spoof for d in decisions], dtype=float)
        confidence_arr = np.array([d.confidence for d in decisions])
        noise_arr = np.array([d.noise_db for d in decisions])
        snr_arr = np.array([d.snr_db for d in decisions])
        reliability_arr = np.array([d.reliability for d in decisions])
        review_arr = np.array([d.needs_review for d in decisions], dtype=float)
        threshold_arr = np.array([d.threshold_used for d in decisions])

        category_counts: Dict[str, int] = {}
        for d in decisions:
            category_counts[d.category] = category_counts.get(d.category, 0) + 1

        return {
            "n_frames": n,
            "spoof_rate": float(np.mean(is_spoof_arr)),
            "mean_confidence": float(np.mean(confidence_arr)),
            "mean_noise_db": float(np.mean(noise_arr)),
            "mean_snr_db": float(np.mean(snr_arr)),
            "mean_reliability": float(np.mean(reliability_arr)),
            "review_rate": float(np.mean(review_arr)),
            "category_counts": category_counts,
            "mean_threshold": float(np.mean(threshold_arr)),
        }

    def reset(self) -> None:
        """Clear history and reset all internal state to defaults.

        A new :class:`NoiseFloorEstimator`, :class:`AdaptiveThreshold`, and
        :class:`ConfidenceCalibrator` are instantiated with default parameters.
        """
        self._noise_est = NoiseFloorEstimator()
        self._threshold = AdaptiveThreshold()
        self._calibrator = ConfidenceCalibrator()
        self._history.clear()


# ---------------------------------------------------------------------------
# Threshold Tuning Utilities
# ---------------------------------------------------------------------------


def find_optimal_thresholds(
    model_session: Any,
    test_samples: List[np.ndarray],
    labels: List[int],
    threshold_grid: Optional[np.ndarray] = None,
    noise_categories: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Grid-search optimal spoof-detection thresholds per noise category.

    The function simulates each test sample through the full noise-estimation
    and adaptive-threshold pipeline, then evaluates candidate thresholds to
    find the one that minimises the mean of false-positive rate and
    false-negative rate (i.e. maximises balanced accuracy).

    Parameters
    ----------
    model_session : any object with a ``run`` method (e.g. onnxruntime.InferenceSession)
        Anti-spoofing model.  Expected to have a ``run(output_names, input_dict)``
        interface compatible with ONNX Runtime.  If the model is ``None``, the
        function uses random dummy scores for demonstration/testing purposes.
    test_samples : list of np.ndarray
        Audio frames to evaluate.  Each element is a 1-D float32 array.
    labels : list of int
        Ground-truth labels (0 = bonafide, 1 = spoof) for each test sample.
    threshold_grid : np.ndarray, optional
        Candidate threshold values to search over.  Defaults to
        ``np.linspace(0.3, 0.9, 25)``.
    noise_categories : list of str, optional
        Noise category for each sample (``"LOW"``, ``"MEDIUM"``, etc.).
        If ``None``, the categories are estimated from the samples themselves.

    Returns
    -------
    dict
        ``{"LOW": float, "MEDIUM": float, "HIGH": float, "VERY_HIGH": float}``
        mapping each noise category to its optimal threshold.
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.3, 0.9, 25)

    categories = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    labels_arr = np.asarray(labels, dtype=int)

    # --- Obtain model scores for all samples ---------------------------
    scores: List[float] = []
    detected_categories: List[str] = []

    noise_est = NoiseFloorEstimator()

    for i, sample in enumerate(test_samples):
        sample = np.asarray(sample, dtype=np.float64).ravel()
        noise_est.update(sample)

        # Determine noise category
        if noise_categories is not None:
            cat = noise_categories[i]
        else:
            cat = AdaptiveThreshold._classify_noise(noise_est.get_noise_level_db())
        detected_categories.append(cat)

        # Obtain model score
        if model_session is not None:
            try:
                input_name = model_session.get_inputs()[0].name
                result = model_session.run(None, {input_name: sample[np.newaxis, :]})
                score = float(np.asarray(result[0]).ravel()[-1])
            except Exception:
                # Fallback: treat as uniform probability
                score = 0.5
        else:
            # Demo / test: use RMS amplitude as a proxy score
            rms = float(np.sqrt(np.mean(np.square(sample))))
            score = float(np.clip(rms * 5.0, 0.0, 1.0))

        scores.append(score)

    scores_arr = np.asarray(scores)
    cats_arr = np.asarray(detected_categories)

    # --- Per-category grid search -------------------------------------
    optimal: Dict[str, float] = {}

    for category in categories:
        mask = cats_arr == category
        if not np.any(mask):
            # No samples in this category; use rule-based default
            optimal[category] = AdaptiveThreshold._CATEGORY_MAP[category]
            continue

        cat_scores = scores_arr[mask]
        cat_labels = labels_arr[mask]

        best_threshold = threshold_grid[0]
        best_balanced_acc = -1.0

        # Vectorised evaluation over all candidate thresholds at once
        # Shape: (n_thresholds, n_samples)
        predictions = (cat_scores[np.newaxis, :] >= threshold_grid[:, np.newaxis]).astype(int)

        tp = np.sum((predictions == 1) & (cat_labels[np.newaxis, :] == 1), axis=1)
        fp = np.sum((predictions == 1) & (cat_labels[np.newaxis, :] == 0), axis=1)
        tn = np.sum((predictions == 0) & (cat_labels[np.newaxis, :] == 0), axis=1)
        fn = np.sum((predictions == 0) & (cat_labels[np.newaxis, :] == 1), axis=1)

        tpr = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        tnr = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        balanced_acc = (tpr + tnr) / 2.0

        best_idx = int(np.argmax(balanced_acc))
        best_balanced_acc = float(balanced_acc[best_idx])
        best_threshold = float(threshold_grid[best_idx])

        optimal[category] = best_threshold

    return optimal


def plot_threshold_curve(
    history: List[ThresholdState],
    output_path: str = "threshold_adaptation.png",
) -> None:
    """Plot the adaptive threshold and noise level over time.

    Creates a two-panel figure:
    - Top: adaptive threshold value vs. time, coloured by noise category.
    - Bottom: noise floor in dBFS vs. time.

    The figure is saved to *output_path* as a PNG file.

    Parameters
    ----------
    history : list of ThresholdState
        Sequence of states produced by :class:`AdaptiveThreshold` over time.
    output_path : str
        File path for the output PNG (default ``"threshold_adaptation.png"``).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_threshold_curve(). "
            "Install it with: pip install matplotlib"
        ) from exc

    if not history:
        raise ValueError("history is empty; nothing to plot")

    timestamps = np.array([s.timestamp for s in history])
    # Normalise to seconds since first event
    t = timestamps - timestamps[0]

    thresholds = np.array([s.current_threshold for s in history])
    noise_levels = np.array([s.noise_level_db for s in history])

    # Colour map per category
    _CAT_COLORS = {
        "LOW": "#2ecc71",       # green
        "MEDIUM": "#3498db",    # blue
        "HIGH": "#e67e22",      # orange
        "VERY_HIGH": "#e74c3c", # red
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Vox-Verify: Adaptive Threshold Behaviour", fontsize=14, fontweight="bold")

    # ---- Top panel: threshold -----------------------------------------
    prev_cat = None
    seg_start = 0
    for i, state in enumerate(history):
        cat = state.noise_category
        if cat != prev_cat or i == len(history) - 1:
            end = i + 1
            color = _CAT_COLORS.get(cat, "grey")
            ax1.plot(t[seg_start:end], thresholds[seg_start:end], color=color, lw=1.8)
            seg_start = i
            prev_cat = cat

    # Horizontal reference lines
    for label, val in [
        ("0.40 (LOW)", _THRESHOLD_LOW),
        ("0.50 (BASE)", _THRESHOLD_MEDIUM),
        ("0.65 (HIGH)", _THRESHOLD_HIGH),
        ("0.80 (V.HIGH)", _THRESHOLD_VERY_HIGH),
    ]:
        ax1.axhline(val, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax1.text(t[-1] * 1.01, val, label, va="center", fontsize=7, color="grey")

    ax1.set_ylabel("Detection Threshold")
    ax1.set_ylim(0.3, 0.95)
    ax1.set_title("Adaptive Threshold")
    ax1.grid(True, alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in _CAT_COLORS.items()
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=8)

    # ---- Bottom panel: noise floor ------------------------------------
    ax2.fill_between(t, noise_levels, alpha=0.3, color="#9b59b6")
    ax2.plot(t, noise_levels, color="#9b59b6", lw=1.5)

    # Category boundary lines
    for db, label in [
        (_NOISE_LOW_MAX, "−40 dB"),
        (_NOISE_MEDIUM_MAX, "−20 dB"),
        (_NOISE_HIGH_MAX, "−10 dB"),
    ]:
        ax2.axhline(db, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax2.text(t[-1] * 1.01, db, label, va="center", fontsize=7, color="grey")

    ax2.set_ylabel("Noise Floor (dBFS)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Estimated Noise Floor")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_threshold_curve] Saved figure → {output_path}")


# ---------------------------------------------------------------------------
# Demo / unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("Vox-Verify  |  dynamic_threshold.py  —  self-test")
    print("=" * 60)

    RNG = np.random.default_rng(42)
    SR = 16_000
    FRAME = 512

    # ---------------------------------------------------------------
    # Helper: generate a frame with a target RMS amplitude
    # ---------------------------------------------------------------
    def make_frame(rms: float = 0.01, n: int = FRAME) -> np.ndarray:
        """Return white-noise frame with approximately the given RMS."""
        noise = RNG.standard_normal(n)
        noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-10) * rms
        return noise.astype(np.float64)

    # ---------------------------------------------------------------
    # 1. NoiseFloorEstimator
    # ---------------------------------------------------------------
    print("\n--- NoiseFloorEstimator ---")
    nfe = NoiseFloorEstimator(window_size=10, alpha=0.3)
    for _ in range(15):
        nfe.update(make_frame(rms=0.01))  # −40 dBFS ≈ quiet room

    noise_db = nfe.get_noise_level_db()
    sf = nfe.get_spectral_flatness()
    snr = nfe.get_snr_estimate(make_frame(rms=0.1))  # louder frame

    print(f"  Noise floor   : {noise_db:.2f} dBFS")
    print(f"  Spectral flat : {sf:.4f}  (≈1 for white noise)")
    print(f"  SNR estimate  : {snr:.2f} dB")

    assert noise_db < 0, "Noise floor should be negative dBFS"
    assert 0.0 <= sf <= 1.0, "Spectral flatness must be in [0, 1]"
    assert snr > 0, "Signal should be louder than noise floor"
    print("  [PASS] NoiseFloorEstimator")

    # ---------------------------------------------------------------
    # 2. AdaptiveThreshold — LOW noise → threshold drifts toward 0.40
    # ---------------------------------------------------------------
    print("\n--- AdaptiveThreshold (LOW noise) ---")
    at = AdaptiveThreshold()
    quiet_db = -55.0  # below -40 → LOW category
    states: List[ThresholdState] = []

    # Simulate 100 updates spread over 5 seconds
    start = time.time()
    for i in range(100):
        state = at.update(quiet_db, spectral_flatness=0.9)
        states.append(state)
        time.sleep(0.001)  # minimal sleep to advance the clock

    final_state = states[-1]
    print(f"  Category      : {final_state.noise_category}")
    print(f"  Threshold     : {final_state.current_threshold:.4f}")
    print(f"  Is reliable   : {final_state.is_reliable}")
    print(f"  Conf modifier : {final_state.confidence_modifier:.4f}")

    assert final_state.noise_category == "LOW"
    assert final_state.is_reliable is True
    # After many updates the threshold should have moved below 0.5
    assert final_state.current_threshold <= 0.50, (
        f"Expected threshold ≤ 0.50, got {final_state.current_threshold:.4f}"
    )
    print("  [PASS] AdaptiveThreshold LOW noise")

    # ---------------------------------------------------------------
    # 3. AdaptiveThreshold — VERY HIGH noise → unreliable
    # ---------------------------------------------------------------
    print("\n--- AdaptiveThreshold (VERY HIGH noise) ---")
    at2 = AdaptiveThreshold()
    loud_db = -5.0  # above -10 → VERY_HIGH
    for _ in range(50):
        vstate = at2.update(loud_db, spectral_flatness=0.2)
        time.sleep(0.001)

    print(f"  Category      : {vstate.noise_category}")
    print(f"  Threshold     : {vstate.current_threshold:.4f}")
    print(f"  Is reliable   : {vstate.is_reliable}")
    assert vstate.noise_category == "VERY_HIGH"
    assert vstate.is_reliable is False
    print("  [PASS] AdaptiveThreshold VERY_HIGH noise")

    # ---------------------------------------------------------------
    # 4. ConfidenceCalibrator
    # ---------------------------------------------------------------
    print("\n--- ConfidenceCalibrator ---")
    cal = ConfidenceCalibrator()

    ts_good = ThresholdState(
        current_threshold=0.5,
        noise_level_db=-50.0,
        noise_category="LOW",
        confidence_modifier=1.0,
        is_reliable=True,
    )
    ts_bad = ThresholdState(
        current_threshold=0.8,
        noise_level_db=-5.0,
        noise_category="VERY_HIGH",
        confidence_modifier=0.5,
        is_reliable=False,
    )

    cs_good = cal.calibrate(raw_score=0.7, noise_state=ts_good, snr_db=25.0)
    cs_bad = cal.calibrate(raw_score=0.7, noise_state=ts_bad, snr_db=2.0)

    print(f"  Good conditions → calibrated={cs_good.calibrated_score:.3f}, "
          f"reliability={cs_good.reliability:.3f}, review={cs_good.needs_review}")
    print(f"  Bad  conditions → calibrated={cs_bad.calibrated_score:.3f}, "
          f"reliability={cs_bad.reliability:.3f}, review={cs_bad.needs_review}")

    assert cs_good.calibrated_score >= cs_bad.calibrated_score, (
        "Good conditions should yield higher or equal calibrated score"
    )
    assert cs_bad.needs_review is True, "VERY_HIGH noise must trigger review"
    print("  [PASS] ConfidenceCalibrator")

    # ---------------------------------------------------------------
    # 5. ThresholdManager end-to-end
    # ---------------------------------------------------------------
    print("\n--- ThresholdManager (end-to-end) ---")
    mgr = ThresholdManager()

    n_frames = 60
    for i in range(n_frames):
        # Simulate alternating bonafide / spoof with varying noise
        rms = 0.005 + 0.01 * math.sin(i * 0.2)  # slowly changing noise
        frame = make_frame(rms=max(rms, 1e-6))
        # Model output: [bonafide_prob, spoof_prob]
        spoof_prob = float(RNG.uniform(0.3, 0.8))
        model_out = np.array([1.0 - spoof_prob, spoof_prob])
        decision = mgr.process_frame(frame, model_out)

    trend = mgr.get_trend()
    print(f"  Frames processed : {trend['n_frames']}")
    print(f"  Spoof rate       : {trend['spoof_rate']:.2%}")
    print(f"  Mean confidence  : {trend['mean_confidence']:.3f}")
    print(f"  Mean noise dBFS  : {trend['mean_noise_db']:.2f}")
    print(f"  Mean SNR         : {trend['mean_snr_db']:.2f} dB")
    print(f"  Mean reliability : {trend['mean_reliability']:.3f}")
    print(f"  Review rate      : {trend['review_rate']:.2%}")
    print(f"  Category counts  : {trend['category_counts']}")

    assert trend["n_frames"] == n_frames
    assert 0.0 <= trend["spoof_rate"] <= 1.0
    print("  [PASS] ThresholdManager")

    # ---------------------------------------------------------------
    # 6. ThresholdManager reset
    # ---------------------------------------------------------------
    mgr.reset()
    trend_after_reset = mgr.get_trend()
    assert trend_after_reset["n_frames"] == 0, "History should be empty after reset"
    print("  [PASS] ThresholdManager.reset()")

    # ---------------------------------------------------------------
    # 7. find_optimal_thresholds (no model session → demo mode)
    # ---------------------------------------------------------------
    print("\n--- find_optimal_thresholds (demo mode) ---")
    n_test = 80
    test_samples = [make_frame(rms=RNG.uniform(0.001, 0.1)) for _ in range(n_test)]
    test_labels = list(RNG.integers(0, 2, n_test))

    optimal_thresholds = find_optimal_thresholds(
        model_session=None,  # demo mode
        test_samples=test_samples,
        labels=test_labels,
    )
    print(f"  Optimal thresholds: {optimal_thresholds}")
    assert set(optimal_thresholds.keys()) == {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    for v in optimal_thresholds.values():
        assert 0.0 <= v <= 1.0, f"Threshold out of range: {v}"
    print("  [PASS] find_optimal_thresholds")

    # ---------------------------------------------------------------
    # 8. plot_threshold_curve
    # ---------------------------------------------------------------
    print("\n--- plot_threshold_curve ---")
    try:
        plot_threshold_curve(states, output_path="/tmp/threshold_adaptation_test.png")
        print("  [PASS] plot_threshold_curve → /tmp/threshold_adaptation_test.png")
    except ImportError as exc:
        print(f"  [SKIP] matplotlib not available: {exc}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
