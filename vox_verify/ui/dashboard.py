"""
Vox-Verify Local – Streamlit Dashboard
=======================================
Real-time deepfake detection status dashboard.

Launch with:
    streamlit run vox_verify/ui/dashboard.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Path bootstrap – allow running from repo root or the ui/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    import numpy as np  # fallback

from vox_verify.ui.logger import DetectionLogger, get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_SAMPLE_RATE = 16_000          # samples per second
_WAVEFORM_SECS = 5             # rolling window
_WAVEFORM_POINTS = 250         # downsampled display points
_REFRESH_INTERVAL = 0.5        # seconds between auto-refresh ticks
_RAM_WARNING_GB = 1.5
_SPOOF_THRESHOLD = 0.5         # score above which we call it synthetic

# Colour palette
_COL_GREEN  = "#22c55e"
_COL_RED    = "#ef4444"
_COL_YELLOW = "#eab308"
_COL_BG     = "#0f1117"
_COL_SURFACE= "#1e2130"
_COL_BORDER = "#2d3148"
_COL_TEXT   = "#e2e8f0"
_COL_MUTED  = "#8892a4"

_DEVICES = ["Default Microphone", "USB Audio Interface", "Built-in Input", "Virtual Cable"]
_MODELS  = {
    "FP32 (Full Precision)": {"tag": "FP32", "size_mb": 94},
    "FP16 (Half Precision)": {"tag": "FP16", "size_mb": 47},
    "INT8 (Quantized)":      {"tag": "INT8", "size_mb": 24},
}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vox-Verify Local",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Base ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #13161f;
    border-right: 1px solid #2d3148;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
}

/* Hide default Streamlit header / footer */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Cards ────────────────────────────────────────────── */
.vv-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.vv-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8892a4;
    margin-bottom: 0.75rem;
}

/* ── Confidence indicator ─────────────────────────────── */
.vv-indicator-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 0;
}

.vv-circle {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    transition: background 0.6s ease, box-shadow 0.6s ease;
}

.vv-circle-safe {
    background: radial-gradient(circle at 40% 35%, #22c55e44, #22c55e11);
    border: 3px solid #22c55e;
    box-shadow: 0 0 40px #22c55e55, 0 0 80px #22c55e22;
}

.vv-circle-spoof {
    background: radial-gradient(circle at 40% 35%, #ef444444, #ef444411);
    border: 3px solid #ef4444;
    box-shadow: 0 0 40px #ef444455, 0 0 80px #ef444422;
}

.vv-circle-uncertain {
    background: radial-gradient(circle at 40% 35%, #eab30844, #eab30811);
    border: 3px solid #eab308;
    box-shadow: 0 0 40px #eab30855, 0 0 80px #eab30822;
}

.vv-circle-idle {
    background: radial-gradient(circle at 40% 35%, #2d314844, #2d314811);
    border: 3px solid #2d3148;
    box-shadow: none;
}

.vv-pct {
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1;
    color: #e2e8f0;
}

.vv-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #b0bec5;
    margin-top: 0.35rem;
    text-align: center;
    padding: 0 0.75rem;
}

/* Pulse animation for active monitoring */
@keyframes vv-pulse {
    0%   { box-shadow: 0 0 40px var(--glow), 0 0 80px var(--glow2); }
    50%  { box-shadow: 0 0 65px var(--glow), 0 0 120px var(--glow2); }
    100% { box-shadow: 0 0 40px var(--glow), 0 0 80px var(--glow2); }
}

.vv-circle-safe.vv-active {
    --glow:  #22c55e88;
    --glow2: #22c55e33;
    animation: vv-pulse 2s ease-in-out infinite;
}
.vv-circle-spoof.vv-active {
    --glow:  #ef444488;
    --glow2: #ef444433;
    animation: vv-pulse 1.2s ease-in-out infinite;
}
.vv-circle-uncertain.vv-active {
    --glow:  #eab30888;
    --glow2: #eab30833;
    animation: vv-pulse 2s ease-in-out infinite;
}

/* ── Status dot ───────────────────────────────────────── */
.vv-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.vv-dot-active   { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.vv-dot-inactive { background: #8892a4; }

/* ── Status bar ───────────────────────────────────────── */
.vv-statusbar {
    display: flex;
    gap: 2rem;
    align-items: center;
    background: #13161f;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 0.6rem 1.25rem;
    font-size: 0.8rem;
    color: #8892a4;
}
.vv-statusbar strong { color: #e2e8f0; }
.vv-statusbar .warn  { color: #f59e0b; }

/* ── Dataframe overrides ──────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation (survives Streamlit reruns)
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "monitoring":       False,
        "waveform_data":    [0.0] * _WAVEFORM_POINTS,
        "detection_history":[], # list[dict] – in-memory cache of recent events
        "engine":           None,  # placeholder for real engine reference
        "logger":           None,
        "last_result":      None,  # most recent detection dict
        "tick_count":       0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Logger is created once
    if st.session_state["logger"] is None:
        st.session_state["logger"] = get_logger(log_dir=_LOG_DIR)

_init_session()

logger: DetectionLogger = st.session_state["logger"]

# ---------------------------------------------------------------------------
# Sidebar – controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.5rem;">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-label="Vox-Verify logo">
              <circle cx="14" cy="14" r="13" stroke="#6366f1" stroke-width="2"/>
              <rect x="10" y="7" width="8" height="12" rx="4" fill="#6366f1"/>
              <path d="M7 16c0 3.866 3.134 7 7 7s7-3.134 7-7" stroke="#a5b4fc" stroke-width="1.8" stroke-linecap="round"/>
              <line x1="14" y1="23" x2="14" y2="26" stroke="#a5b4fc" stroke-width="1.8" stroke-linecap="round"/>
              <line x1="10" y1="26" x2="18" y2="26" stroke="#a5b4fc" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;letter-spacing:0.02em;">
                Vox-Verify <span style="color:#6366f1;">Local</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Controls")

    # Start / Stop
    monitoring = st.session_state["monitoring"]
    if monitoring:
        if st.button("⏹  Stop Monitoring", type="secondary", use_container_width=True):
            st.session_state["monitoring"] = False
            st.session_state["last_result"] = None
            st.rerun()
    else:
        if st.button("▶  Start Monitoring", type="primary", use_container_width=True):
            st.session_state["monitoring"] = True
            st.rerun()

    st.divider()

    st.markdown("#### Configuration")

    device = st.selectbox(
        "Audio Device",
        options=_DEVICES,
        index=0,
        help="Select the microphone or audio input device.",
    )

    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher sensitivity → flag borderline cases as spoof.",
    )

    model_key = st.selectbox(
        "Model Precision",
        options=list(_MODELS.keys()),
        index=0,
        help="FP32 = most accurate, INT8 = fastest / smallest RAM.",
    )
    model_info = _MODELS[model_key]

    st.divider()
    st.markdown(
        f"<span style='font-size:0.75rem;color:#8892a4;'>Model: <strong style='color:#e2e8f0;'>"
        f"{model_info['tag']}</strong> · {model_info['size_mb']} MB</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Simulation helpers (replace with real engine calls when engine exists)
# ---------------------------------------------------------------------------
def _simulate_detection(sensitivity: float) -> dict:
    """Generate a plausible detection result for demo / testing."""
    is_spoof = random.random() < 0.15  # 15% chance of spoof
    if is_spoof:
        spoof_score   = random.uniform(0.55, 0.98)
        bonafide_score = 1.0 - spoof_score
    else:
        bonafide_score = random.uniform(0.70, 0.99)
        spoof_score    = 1.0 - bonafide_score

    # Sensitivity shifts the effective threshold
    eff_threshold = _SPOOF_THRESHOLD - (sensitivity - 0.5) * 0.3
    is_spoof_call = spoof_score > max(0.1, eff_threshold)

    confidence = max(bonafide_score, spoof_score)
    return {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "bonafide_score":  round(bonafide_score, 4),
        "spoof_score":     round(spoof_score, 4),
        "is_spoof":        is_spoof_call,
        "confidence":      round(confidence, 4),
        "noise_level_db":  round(random.uniform(-60, -20), 1),
        "model_latency_ms":round(random.uniform(18, 95), 1),
    }


def _simulate_waveform_chunk(n: int = 20) -> list[float]:
    """Simulate a small chunk of audio amplitude values."""
    base = random.uniform(0.01, 0.08)
    return [base + random.gauss(0, 0.05) for _ in range(n)]


def _update_waveform(chunk: list[float]) -> None:
    wf = st.session_state["waveform_data"]
    wf.extend(chunk)
    st.session_state["waveform_data"] = wf[-_WAVEFORM_POINTS:]

# ---------------------------------------------------------------------------
# Confidence indicator HTML builder
# ---------------------------------------------------------------------------
def _confidence_html(result: dict | None, active: bool) -> str:
    if result is None:
        return """
        <div class="vv-indicator-wrap">
            <div class="vv-circle vv-circle-idle">
                <span class="vv-pct" style="color:#8892a4;">—</span>
                <span class="vv-label">Not monitoring</span>
            </div>
        </div>
        """

    bonafide = result["bonafide_score"]
    spoof    = result["spoof_score"]
    conf     = result["confidence"]
    pct_str  = f"{conf * 100:.1f}%"

    diff = abs(bonafide - spoof)
    if diff < 0.15:
        css_cls = "vv-circle-uncertain"
        label   = "Uncertain"
        colour  = _COL_YELLOW
    elif bonafide > spoof:
        css_cls = "vv-circle-safe"
        label   = "Authentic"
        colour  = _COL_GREEN
    else:
        css_cls = "vv-circle-spoof"
        label   = "Synthetic Detected"
        colour  = _COL_RED

    active_cls = "vv-active" if active else ""

    return f"""
    <div class="vv-indicator-wrap">
        <div class="vv-circle {css_cls} {active_cls}">
            <span class="vv-pct" style="color:{colour};">{pct_str}</span>
            <span class="vv-label">{label}</span>
        </div>
        <div style="margin-top:1rem;display:flex;gap:1.5rem;font-size:0.8rem;color:#8892a4;">
            <span>Bonafide <strong style="color:#22c55e;">{bonafide*100:.1f}%</strong></span>
            <span>Spoof <strong style="color:#ef4444;">{spoof*100:.1f}%</strong></span>
        </div>
    </div>
    """

# ---------------------------------------------------------------------------
# Waveform chart
# ---------------------------------------------------------------------------
def _render_waveform(waveform: list[float], events: list[dict]) -> None:
    times = [i / (_WAVEFORM_POINTS / _WAVEFORM_SECS) for i in range(len(waveform))]

    if _HAS_PLOTLY:
        fig = go.Figure()

        # Waveform trace
        fig.add_trace(go.Scatter(
            x=times, y=waveform,
            mode="lines",
            line=dict(color="#6366f1", width=1.5),
            name="Amplitude",
            hovertemplate="t=%{x:.2f}s  amp=%{y:.3f}<extra></extra>",
        ))

        # Overlay spoof events as vertical markers
        for ev in events[-10:]:
            if ev.get("is_spoof"):
                # Map event to a rough x position (last events → right side)
                x_pos = _WAVEFORM_SECS - 0.2
                fig.add_vline(
                    x=x_pos,
                    line_color=_COL_RED,
                    line_dash="dot",
                    line_width=1,
                    annotation_text="⚠",
                    annotation_font_color=_COL_RED,
                    annotation_font_size=14,
                )

        fig.update_layout(
            paper_bgcolor=_COL_BG,
            plot_bgcolor=_COL_SURFACE,
            font=dict(color=_COL_TEXT, size=11),
            margin=dict(l=10, r=10, t=10, b=30),
            height=180,
            xaxis=dict(
                title="Time (s)",
                range=[0, _WAVEFORM_SECS],
                gridcolor=_COL_BORDER,
                zeroline=False,
                tickcolor=_COL_MUTED,
            ),
            yaxis=dict(
                title="Amplitude",
                gridcolor=_COL_BORDER,
                zeroline=True,
                zerolinecolor=_COL_BORDER,
                tickcolor=_COL_MUTED,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # Fallback: built-in line chart
        import pandas as pd
        df = pd.DataFrame({"amplitude": waveform}, index=times)
        st.line_chart(df, height=180, use_container_width=True)

# ---------------------------------------------------------------------------
# Detection log table
# ---------------------------------------------------------------------------
def _render_log_table(events: list[dict]) -> None:
    if not events:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.85rem;'>No detections yet.</p>",
            unsafe_allow_html=True,
        )
        return

    import pandas as pd

    rows = []
    for ev in reversed(events[-100:]):
        ts = ev.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_fmt = dt.strftime("%H:%M:%S")
        except Exception:
            ts_fmt = ts[:19] if len(ts) >= 19 else ts

        rows.append({
            "Time":        ts_fmt,
            "Bonafide %":  f"{ev.get('bonafide_score', 0)*100:.1f}",
            "Spoof %":     f"{ev.get('spoof_score', 0)*100:.1f}",
            "Result":      "⚠ Synthetic" if ev.get("is_spoof") else "✓ Authentic",
            "Confidence":  f"{ev.get('confidence', 0)*100:.1f}%",
            "Noise dB":    f"{ev.get('noise_level_db', 0):.1f}",
            "Latency ms":  f"{ev.get('model_latency_ms', 0):.1f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, 50 + len(rows) * 38),
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------
def _render_status_bar(device: str, model_info: dict) -> None:
    latency_str = "—"
    hist = st.session_state["detection_history"]
    if hist:
        latency_str = f"{hist[-1].get('model_latency_ms', 0):.1f} ms"

    ram_str   = "—"
    ram_warn  = False
    if _HAS_PSUTIL:
        proc      = psutil.Process(os.getpid())
        ram_mb    = proc.memory_info().rss / 1024 / 1024
        ram_gb    = ram_mb / 1024
        ram_str   = f"{ram_mb:.0f} MB"
        ram_warn  = ram_gb > _RAM_WARNING_GB

    dot_cls = "vv-dot-active" if st.session_state["monitoring"] else "vv-dot-inactive"
    status_text = "Monitoring" if st.session_state["monitoring"] else "Idle"

    warn_html = (
        "<span class='warn'>⚠ High RAM</span>" if ram_warn else ""
    )
    ram_class = "warn" if ram_warn else ""

    st.markdown(
        f"""
        <div class="vv-statusbar">
            <span>
                <span class="vv-status-dot {dot_cls}"></span>
                <strong>{status_text}</strong>
            </span>
            <span>Device: <strong>{device}</strong></span>
            <span>Model: <strong>{model_info['tag']}</strong> ({model_info['size_mb']} MB)</span>
            <span>Latency: <strong>{latency_str}</strong></span>
            <span class="{ram_class}">RAM: <strong>{ram_str}</strong> {warn_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main page layout
# ---------------------------------------------------------------------------
st.markdown(
    "<h2 style='color:#e2e8f0;margin-bottom:0.25rem;'>Vox-Verify Local</h2>"
    "<p style='color:#8892a4;font-size:0.9rem;margin-top:0;margin-bottom:1.25rem;'>"
    "Real-time deepfake voice detection</p>",
    unsafe_allow_html=True,
)

# Status bar – top of main area
_render_status_bar(device, model_info)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# ── Row 1: Confidence indicator ────────────────────────────────────────────
st.markdown("<div class='vv-card'><div class='vv-card-title'>Detection Status</div>", unsafe_allow_html=True)
confidence_slot = st.empty()
st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Waveform ────────────────────────────────────────────────────────
st.markdown("<div class='vv-card'><div class='vv-card-title'>Live Waveform — last 5 s</div>", unsafe_allow_html=True)
waveform_slot = st.empty()
st.markdown("</div>", unsafe_allow_html=True)

# ── Row 3: Detection log ───────────────────────────────────────────────────
st.markdown("<div class='vv-card'>", unsafe_allow_html=True)
log_col1, log_col2 = st.columns([6, 1])
with log_col1:
    st.markdown("<div class='vv-card-title'>Detection Log</div>", unsafe_allow_html=True)
with log_col2:
    export_btn = st.button("⬇ Export", key="export_btn", help="Download JSON log")
st.markdown("</div>", unsafe_allow_html=True)

log_slot = st.empty()

# Export handler
if export_btn:
    try:
        log_path = logger.current_path
        if log_path.exists() and log_path.stat().st_size > 0:
            with open(log_path, "rb") as fh:
                st.download_button(
                    label="Download detections.json",
                    data=fh.read(),
                    file_name="detections.json",
                    mime="application/json",
                    key="dl_btn",
                )
        else:
            st.info("Log is empty – no events to export yet.")
    except Exception as exc:
        st.warning(f"Export error: {exc}")

# ---------------------------------------------------------------------------
# Real-time refresh loop (only runs when monitoring is active)
# ---------------------------------------------------------------------------
def _do_one_tick() -> None:
    """Simulate one detection tick, update state and log."""
    result = _simulate_detection(sensitivity)
    logger.log_event(result)

    history = st.session_state["detection_history"]
    history.append(result)
    if len(history) > 200:
        st.session_state["detection_history"] = history[-200:]

    st.session_state["last_result"] = result
    st.session_state["tick_count"] += 1

    chunk = _simulate_waveform_chunk(20)
    _update_waveform(chunk)


if st.session_state["monitoring"]:
    _do_one_tick()

# ── Render confidence indicator ───────────────────────────────────────────
with confidence_slot:
    st.markdown(
        _confidence_html(
            st.session_state["last_result"],
            st.session_state["monitoring"],
        ),
        unsafe_allow_html=True,
    )

# ── Render waveform ───────────────────────────────────────────────────────
with waveform_slot:
    _render_waveform(
        st.session_state["waveform_data"],
        st.session_state["detection_history"],
    )

# ── Render log ────────────────────────────────────────────────────────────
# Show from in-memory cache first; fall back to disk for historical records
history = st.session_state["detection_history"]
if not history:
    history = logger.get_recent(50)

with log_slot:
    _render_log_table(history)

# ── Auto-refresh while monitoring ─────────────────────────────────────────
if st.session_state["monitoring"]:
    time.sleep(_REFRESH_INTERVAL)
    st.rerun()
