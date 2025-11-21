"""
Simple command-line image generator using HuggingFace diffusers.
Usage:
  python generate_image.py --prompt "an imaginative scene" --out outputs/out.png --num_inference_steps 30
"""
import argparse
import os
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (M1/M2/M3/M4)
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/generated.png")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading model {args.model} on device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    print("Generating image...")
    image = pipe(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width
    ).images[0]

    image.save(args.out)
    print(f"Saved image to {args.out}")


if __name__ == "__main__":
    main()
