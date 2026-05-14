# vox-verify
WIP Local-first audio deepfake detector built on AASIST-Tiny (88K parameter sparse graph attention network). Runs entirely on-device via ONNX Runtime at sub-15ms latency with no cloud dependency. Includes a real-time Streamlit dashboard, dynamic noise-adaptive thresholding, INT8 quantization, and a 50-sample adversarial stress test suite.
