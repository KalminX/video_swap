"""Curated style transformation presets for Generative AI identity synthesis."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StylePreset:
    name: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    strength: float  # Inpainting denoise strength (0.6 - 0.9)


STYLE_PRESETS: Dict[str, StylePreset] = {
    "cyborg_robot": StylePreset(
        name="Cybernetic Android",
        prompt=(
            "high-tech futuristic cybernetic cyborg android face, glowing blue fiber-optic circuitry under chrome skin, "
            "metallic plating, polished carbon fiber contours, robotic synthetic facial features, photorealistic, 8k, cinematic lighting"
        ),
        negative_prompt=(
            "human skin, blurry, distorted, deformed, cartoon, low quality, bad anatomy, bad eyes, extra limbs"
        ),
        guidance_scale=8.0,
        strength=0.78,
    ),
    "extraterrestrial_alien": StylePreset(
        name="Extraterrestrial Alien",
        prompt=(
            "celestial alien humanoid with iridescent bioluminescent skin, subtle organic head ridges, deep obsidian cosmic eyes, "
            "delicate extraterrestrial facial geometry, ethereal sci-fi aesthetic, hyper-detailed, masterpiece"
        ),
        negative_prompt=(
            "monstrous, grotesque, blurry, messy, human, lowres, oversaturated, deformed"
        ),
        guidance_scale=8.5,
        strength=0.82,
    ),
    "marble_statue": StylePreset(
        name="Classical Greco-Roman Statue",
        prompt=(
            "classical Greco-Roman marble sculpture portrait, carved white Carrara marble with fine mineral veins, "
            "sculpted smooth facial features, museum lighting, chisel texture, antique masterpiece, high dynamic range"
        ),
        negative_prompt=(
            "flesh, skin, modern clothes, colorful, plastic, blurry, low resolution, bad proportions"
        ),
        guidance_scale=7.5,
        strength=0.80,
    ),
    "watercolor_anime": StylePreset(
        name="Watercolor Anime Illustration",
        prompt=(
            "delicate watercolor and ink illustration of a fantasy character face, soft pastel color wash, clean anime line art, "
            "expressive eyes, Studio Ghibli style, paper texture, masterpiece, artistic"
        ),
        negative_prompt=(
            "photograph, 3d render, plastic, muddy colors, harsh shadows, low quality, artifacts"
        ),
        guidance_scale=7.5,
        strength=0.75,
    ),
    "steampunk_automaton": StylePreset(
        name="Steampunk Mechanoid",
        prompt=(
            "intricate Victorian steampunk automaton face, brass gears and clockwork mechanisms embedded in porcelain skin, "
            "copper riveting, ornate filigree details, warm amber atmospheric lighting, 8k resolution"
        ),
        negative_prompt=(
            "modern electronics, neon, blurry, plastic, lowres, bad anatomy, deformed"
        ),
        guidance_scale=8.0,
        strength=0.80,
    ),
}


def get_preset(name: str) -> StylePreset:
    """Retrieve a style preset by key, defaulting to cyborg_robot if not found."""
    return STYLE_PRESETS.get(name, STYLE_PRESETS["cyborg_robot"])


def list_available_presets() -> List[str]:
    """List all available transformation preset identifiers."""
    return list(STYLE_PRESETS.keys())
