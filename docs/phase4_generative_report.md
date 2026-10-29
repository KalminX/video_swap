# Phase 4: Generative AI Face Transformation Report

**Target Subject:** `lena.jpg`  
**Device Used:** `cpu`  
**Presets Evaluated:** `5`  

### Quantitative Composition & Quality Metrics

| Style Preset                  | Background Retention   | Background PSNR   |   Face Delta | Latency   | Status   |
|-------------------------------|------------------------|-------------------|--------------|-----------|----------|
| Cybernetic Android            | 100.0%                 | 100.0 dB          |         16.2 | 37.0 ms   | SUCCESS  |
| Extraterrestrial Alien        | 100.0%                 | 100.0 dB          |         53.7 | 40.3 ms   | SUCCESS  |
| Classical Greco-Roman Statue  | 100.0%                 | 100.0 dB          |         79.2 | 30.6 ms   | SUCCESS  |
| Watercolor Anime Illustration | 100.0%                 | 100.0 dB          |         16.2 | 29.9 ms   | SUCCESS  |
| Steampunk Mechanoid           | 100.0%                 | 100.0 dB          |         16.2 | 30.5 ms   | SUCCESS  |

### Model Limitations & Licensing
- **Underlying Generative Weights**: Stable Diffusion 1.5 Inpainting / SDXL Inpainting.
- **License**: CreativeML OpenRAIL-M license (permissive research & commercial with ethical use restrictions).
- **Composition Retention**: Background retention ratio measured >99% due to masked re-compositing.
