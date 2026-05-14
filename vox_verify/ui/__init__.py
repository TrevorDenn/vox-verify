"""
vox_verify.ui
=============
Streamlit dashboard and supporting utilities for Vox-Verify Local.

Modules
-------
dashboard   – Streamlit app entry point
logger      – Thread-safe JSON-Lines detection event logger
"""

from vox_verify.ui.logger import DetectionLogger, get_logger

__all__ = ["DetectionLogger", "get_logger"]
