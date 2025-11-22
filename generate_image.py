"""
Simple command-line image generator using HuggingFace diffusers.
Usage:
  python generate_image.py --prompt "an imaginative scene" --out outputs/out.png --num_inference_steps 30
"""
import argparse
from diffusers import AutoPipelineForText2Image
import torch
import os

def generate_image(prompt, output_path):
    print("Loading SD-Turbo model (this is small & fast)...")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    # Use Metal GPU (MPS) on Mac
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    pipe = pipe.to(device)

    print(f"Using device: {device}")
    print(f"Generating image for prompt: {prompt}")

    # Turbo = only 1–4 inference steps needed
    image = pipe(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=0.0
    ).images[0]

    # Create output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    print(f"Image saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/image.png")
    args = parser.parse_args()

    generate_image(args.prompt, args.out)
