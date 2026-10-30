# Google Colab GPU Setup — Copy-Paste Cells

Each section below represents a single cell in your Google Colab notebook. Copy each block into its own notebook cell in order.

Make sure your Colab notebook runtime is set to **GPU**:  
`Runtime -> Change runtime type -> T4 GPU (or A100)`

---

### Cell 1: Check GPU
```bash
!nvidia-smi
```

---

### Cell 2: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/KalminX/video_swap.git
%cd video_swap

!pip install -q diffusers transformers accelerate insightface onnxruntime-gpu mediapipe opencv-python gradio Pillow tabulate
```

---

### Cell 3: Download Model Weights (InSwapper-128 & GFPGAN)
```bash
!mkdir -p models/weights
!curl -L -s -o models/weights/inswapper_128.onnx "https://huggingface.co/ezioroot/inswapper/resolve/main/inswapper_128.onnx"
!curl -L -s -o models/weights/GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
!ls -lh models/weights/
```

---

### Cell 4: Initialize InsightFace & InSwapper on GPU
```python
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image

print("Loading InsightFace Face Analysis (buffalo_l)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("Loading InSwapper-128 on GPU...")
swapper = insightface.model_zoo.get_model(
    "models/weights/inswapper_128.onnx",
    download=False,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("InSwapper model ready for inference!")
```

---

### Cell 5: Initialize Stable Diffusion Inpainting in FP16
```python
import torch
from diffusers import StableDiffusionInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Stable Diffusion Inpainting on {device}...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_attention_slicing()

print("Stable Diffusion pipeline ready!")
```

---

### Cell 6: Facial Inpainting Mask Function
```python
import mediapipe as mp

def extract_face_mask(image_bgr, dilate_px=15, feather_k=17):
    """Generates an automated face boundary mask using 468-point MediaPipe mesh."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    mp_mesh = mp.solutions.face_mesh
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3) as mesh:
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            raise ValueError("No face detected in target image.")
        pts = [[int(min(lm.x * w, w - 1)), int(min(lm.y * h, h - 1))] for lm in res.multi_face_landmarks[0].landmark]
        pts = np.array(pts, dtype=np.int32)
        
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        
    if feather_k > 1:
        mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)
        
    return mask

print("Mask extraction function ready.")
```

---

### Cell 7: Test InSwapper Face Swap
```python
import matplotlib.pyplot as plt

# Load sample faces from the repository
src_bgr = cv2.imread("data/inputs/lena.jpg")
tgt_bgr = cv2.imread("data/inputs/david1.jpg")

# Detect faces and extract identity embeddings
src_faces = face_app.get(src_bgr)
tgt_faces = face_app.get(tgt_bgr)

if src_faces and tgt_faces:
    # Perform single-shot latent face swap
    swapped_bgr = tgt_bgr.copy()
    swapped_bgr = swapper.get(swapped_bgr, tgt_faces[0], src_faces[0], paste_back=True)
    
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Source Identity (Lena)")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Target Scene (David)")
    axes[1].axis("off")
    
    axes[2].imshow(cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Photorealistic Swapped Result")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
else:
    print("Could not detect face in sample images.")
```

---

### Cell 8: Test Generative AI Character Transformation
```python
target_bgr = cv2.imread("data/inputs/lena.jpg")
h, w = target_bgr.shape[:2]

# Generate mask
mask = extract_face_mask(target_bgr)
pil_mask = Image.fromarray(mask)
pil_target = Image.fromarray(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))

# Prompt
prompt = "high-tech futuristic cybernetic cyborg android face, glowing blue circuitry under chrome skin, metallic plating, 8k, cinematic"
neg_prompt = "human skin, blurry, cartoon, low quality, deformed, extra limbs"

output_pil = pipe(
    prompt=prompt,
    negative_prompt=neg_prompt,
    image=pil_target,
    mask_image=pil_mask,
    guidance_scale=8.0,
    strength=0.80,
).images[0]

# Composite with exact background
raw_out = cv2.cvtColor(np.array(output_pil.resize((w, h))), cv2.COLOR_RGB2BGR)
soft_m = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
soft_m_3ch = cv2.merge([soft_m, soft_m, soft_m])
final_bgr = raw_out.astype(np.float32) * soft_m_3ch + target_bgr.astype(np.float32) * (1.0 - soft_m_3ch)
final_bgr = np.clip(final_bgr, 0, 255).astype(np.uint8)

# Display result
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Input")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Facial Mask")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
axes[2].set_title("Transformed Cyborg")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

---

### Cell 9: Launch Gradio Web App with Public Share Link
```python
import gradio as gr

STYLE_PRESETS = {
    "Cybernetic Cyborg": "high-tech futuristic cybernetic cyborg android face, glowing blue circuitry under chrome skin, metallic plating, 8k, cinematic",
    "Extraterrestrial Alien": "celestial alien humanoid with iridescent bioluminescent skin, deep cosmic eyes, delicate extraterrestrial features, masterpiece",
    "Marble Statue": "classical Greco-Roman marble sculpture portrait, carved white Carrara marble with fine mineral veins, museum lighting, antique masterpiece",
    "Watercolor Anime": "delicate watercolor and ink illustration of a fantasy character face, soft pastel color wash, clean anime line art, Studio Ghibli style",
    "Steampunk Mechanoid": "Victorian steampunk automaton face, brass gears and clockwork mechanisms embedded in porcelain skin, ornate filigree, 8k",
}

def gradio_neural_swap(source_img, target_img):
    src_bgr = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    src_faces = face_app.get(src_bgr)
    tgt_faces = face_app.get(tgt_bgr)
    
    if not src_faces:
        raise gr.Error("No face found in Source Image.")
    if not tgt_faces:
        raise gr.Error("No face found in Target Image.")
        
    res_bgr = tgt_bgr.copy()
    for tgt_face in tgt_faces:
        res_bgr = swapper.get(res_bgr, tgt_face, src_faces[0], paste_back=True)
        
    return Image.fromarray(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB))

def gradio_generative(target_img, style_name, custom_prompt, strength):
    tgt_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    h, w = tgt_bgr.shape[:2]
    
    mask = extract_face_mask(tgt_bgr)
    pil_mask = Image.fromarray(mask)
    pil_target = Image.fromarray(cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB))
    
    prompt = custom_prompt if custom_prompt.strip() else STYLE_PRESETS.get(style_name, STYLE_PRESETS["Cybernetic Cyborg"])
    neg_prompt = "monstrous, grotesque, blurry, messy, human, lowres, deformed, bad anatomy, extra limbs"
    
    out_pil = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=pil_target,
        mask_image=pil_mask,
        guidance_scale=8.0,
        strength=strength,
    ).images[0]
    
    raw_out = cv2.cvtColor(np.array(out_pil.resize((w, h))), cv2.COLOR_RGB2BGR)
    soft_m = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
    soft_m_3ch = cv2.merge([soft_m, soft_m, soft_m])
    final_bgr = raw_out.astype(np.float32) * soft_m_3ch + tgt_bgr.astype(np.float32) * (1.0 - soft_m_3ch)
    final_bgr = np.clip(final_bgr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

with gr.Blocks(title="AI Identity Transformation Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 AI Identity Transformation Studio")
    gr.Markdown("**High-fidelity Face Swapping and Generative Character Transformation**")
    
    with gr.Tab("⚡ Photorealistic Neural Face Swap (InSwapper)"):
        with gr.Row():
            with gr.Column():
                src_in = gr.Image(label="Source Identity Image", type="pil")
                tgt_in = gr.Image(label="Target Scene Image", type="pil")
                swap_btn = gr.Button("Execute Neural Swap", variant="primary")
            with gr.Column():
                swap_out = gr.Image(label="Swapped Output", type="pil")
        swap_btn.click(fn=gradio_neural_swap, inputs=[src_in, tgt_in], outputs=swap_out)
        
    with gr.Tab("🤖 Generative Character Transformation"):
        with gr.Row():
            with gr.Column():
                gen_in = gr.Image(label="Target Portrait Image", type="pil")
                preset_dd = gr.Dropdown(choices=list(STYLE_PRESETS.keys()), value="Cybernetic Cyborg", label="Character Preset")
                prompt_txt = gr.Textbox(label="Custom Prompt (Optional)", placeholder="Leave blank to use selected preset")
                denoise_sld = gr.Slider(minimum=0.5, maximum=0.95, value=0.80, step=0.05, label="Transformation Strength")
                gen_btn = gr.Button("Generate Transformation", variant="primary")
            with gr.Column():
                gen_out = gr.Image(label="Transformed Output", type="pil")
        gen_btn.click(fn=gradio_generative, inputs=[gen_in, preset_dd, prompt_txt, denoise_sld], outputs=gen_out)

demo.launch(share=True, debug=False)
```
