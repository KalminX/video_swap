# Phase 1: Benchmark and Verification Report

**Detector Used:** `OpenCV-DNN-YuNet`  
**Total Images Processed:** `5`  

### Latency & Detection Summary

| Image File         | Dimensions   |   Faces | Detect Latency   | Total Latency   | Status   |
|--------------------|--------------|---------|------------------|-----------------|----------|
| baboon.jpg         | 512x512      |       0 | 38.57 ms         | 54.52 ms        | NO FACES |
| chicky_no_face.png | 512x512      |       0 | 12.46 ms         | 29.95 ms        | NO FACES |
| fruits_no_face.jpg | 512x480      |       0 | 20.78 ms         | 31.03 ms        | NO FACES |
| lena.jpg           | 512x512      |       1 | 15.98 ms         | 27.46 ms        | SUCCESS  |
| messi_face.jpg     | 548x342      |       1 | 16.39 ms         | 25.08 ms        | SUCCESS  |

### Completion Criteria Verification
- [x] Successfully processed multiple test images.
- [x] Handled images with no detectable faces without crash.
- [x] Recorded accurate per-stage execution times and diagnostic outputs.
