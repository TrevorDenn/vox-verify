"""
vox_verify.tests
================

Test suite for the VoxVerify anti-spoofing / deepfake-detection pipeline.

Submodules
----------
adversarial_stress_test
    Generates 50 synthetic audio samples across five categories
    (clean speech-like, synthetic artifacts, compression variants,
    noise-corrupted, and edge cases) and runs them through the
    VoxVerify inference model to evaluate robustness.
"""
