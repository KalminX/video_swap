# Phase 3: Classical Face Swapping Benchmark Report

**Blending Method:** `Poisson Seamless Cloning (normal)`  
**Color Matching:** `Reinhard CIELAB Transfer`  
**Total Pairs Tested:** `3`  

### Benchmark Results

| Pair (Source -> Target)   |   Delaunay Triangles | Warp Latency   | Blend Latency   | Total Latency   | Status   |
|---------------------------|----------------------|----------------|-----------------|-----------------|----------|
| lena.jpg -> david1.jpg    |                  229 | 28.4 ms        | 1.9 ms          | 257.5 ms        | SUCCESS  |
| david1.jpg -> lena.jpg    |                  243 | 31.6 ms        | 16.6 ms         | 114.2 ms        | SUCCESS  |
| david1.jpg -> david2.jpg  |                  244 | 28.0 ms        | 5.2 ms          | 82.2 ms         | SUCCESS  |

### Completion Criteria Verification
- [x] Successfully processed multiple test image face swaps.
- [x] Produced aligned facial replacement via Delaunay piecewise affine warping.
- [x] Minimized visible blending boundaries via feathered mask and Poisson seamless cloning.
- [x] Documented limitations involving lighting, occlusion, and head rotation.
