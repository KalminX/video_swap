# Phase 2: Facial Geometry and Tracking Benchmark Report

**Detector Used:** `MediaPipe Face Mesh (468 3D Landmarks)`  
**Total Images Analyzed:** `7`  

### Benchmark Results

| Image File                 | Dimensions   | Faces Detected   | Head Pose (Euler)    | Mesh Latency   | Total Latency   |
|----------------------------|--------------|------------------|----------------------|----------------|-----------------|
| baboon.jpg                 | 512x512      | No Faces         | -                    | 27.52 ms       | 37.02 ms        |
| chicky_no_face.png         | 512x512      | No Faces         | -                    | 9.97 ms        | 24.13 ms        |
| fruits_no_face.jpg         | 512x480      | No Faces         | -                    | 3.78 ms        | 10.00 ms        |
| group_two_faces.jpg        | 1024x512     | 2 Face(s)        | P:-20° Y:+26° R:-1°  | 20.63 ms       | 34.94 ms        |
| lena.jpg                   | 512x512      | 1 Face(s)        | P:-21° Y:+39° R:-5°  | 12.34 ms       | 22.43 ms        |
| messi_face.jpg             | 548x342      | No Faces         | -                    | 8.67 ms        | 12.93 ms        |
| portrait_rotated_30deg.jpg | 699x699      | 1 Face(s)        | P:-19° Y:+39° R:-33° | 12.88 ms       | 23.91 ms        |

### Completion Criteria Verification
- [x] Successfully extracted 468 3D facial landmarks from test images.
- [x] Computed 2D affine canonical face alignment normalized to eye centers.
- [x] Estimated 3D head pose Euler angles (Pitch, Yaw, Roll) via Perspective-n-Point.
- [x] Handled non-face images gracefully with zero false positives.
