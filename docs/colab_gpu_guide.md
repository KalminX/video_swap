# Guide: Running on Google Colab GPU

This guide explains how to run the high-powered **AI Identity Transformation Studio** on a free cloud GPU via Google Colab.

---

## 1. Quick Launch Options

### Option A: Open Directly in Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** tab and upload:
   [`notebooks/AI_Identity_Transformation_Studio_Colab_GPU.ipynb`](file:///Users/kalmin/projects/video_swap/notebooks/AI_Identity_Transformation_Studio_Colab_GPU.ipynb)

### Option B: Push to GitHub & 1-Click Launch
Once your repository is pushed to GitHub, you can open it directly via:
```
https://colab.research.google.com/github/<YOUR_GITHUB_USERNAME>/video_swap/blob/master/notebooks/AI_Identity_Transformation_Studio_Colab_GPU.ipynb
```

---

## 2. Enable GPU Acceleration
In Colab:
1. Click **Runtime** in the top navigation bar.
2. Select **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or A100/V100 if Colab Pro).
4. Click **Save**.

---

## 3. Execution Workflow in Colab

1. **Cell 1 (`!nvidia-smi`)**: Verifies your NVIDIA CUDA GPU is active.
2. **Cell 2 (`pip install ...`)**: Automatically downloads and installs Diffusers, InsightFace, ONNXRuntime-GPU, PyTorch, MediaPipe, and Gradio.
3. **Cell 3 (`Models Setup`)**:
   - Downloads `inswapper_128.onnx` and `GFPGANv1.4.pth`.
   - Initializes InsightFace ArcFace 512-D identity extractor.
   - Loads Stable Diffusion in FP16 with attention slicing for fast generation.
4. **Cell 4 (`Transformation Functions`)**: Defines the neural swap and generative character pipelines.
5. **Cell 5 (`Gradio UI`)**: Launches an interactive web interface with two tabs:
   - **Tab 1: Photorealistic Neural Face Swap (InSwapper)**: Upload any source and target image, and click "Execute Photorealistic Swap".
   - **Tab 2: Generative AI Character Transformation**: Select a preset (Cyborg, Alien, Marble Statue, Anime, Steampunk) or enter a custom prompt.
   - Includes a **public shareable link** (`https://xxxx.gradio.live`) so you can test it from any browser or phone!
