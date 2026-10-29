# Phase 3: Classical Face Swapping Benchmark Report

**Blending Method:** `Poisson Seamless Cloning (normal)`  
**Color Matching:** `Reinhard CIELAB Transfer`  
**Total Pairs Tested:** `3`  

### Benchmark Results

| Pair (Source -> Target)   |   Delaunay Triangles | Warp Latency   | Blend Latency   | Total Latency   | Status   |
|---------------------------|----------------------|----------------|-----------------|-----------------|----------|
| lena.jpg -> david1.jpg    |                  229 | 28.1 ms        | 2.3 ms          | 275.5 ms        | SUCCESS  |
| david1.jpg -> lena.jpg    |                  243 | 36.4 ms        | 19.6 ms         | 124.9 ms        | SUCCESS  |
| david1.jpg -> david2.jpg  |                  244 | 29.0 ms        | 4.0 ms          | 82.3 ms         | SUCCESS  |

### Completion Criteria Verification
- [x] Successfully processed multiple test image face swaps.
- [x] Produced aligned facial replacement via Delaunay piecewise affine warping.
- [x] Minimized visible blending boundaries via feathered mask and Poisson seamless cloning.
- [x] Documented limitations involving lighting, occlusion, and head rotation.
