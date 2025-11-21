"""
Run: python app.py
Open http://localhost:7860
"""
import gradio as gr
from generate_image import get_device
from diffusers import StableDiffusionPipeline
import torch


MODEL_ID = "runwayml/stable-diffusion-v1-5"


# Lazy load pipeline
pipe = None


def load_pipe(device=None):
global pipe
if pipe is None:
dev = device or get_device()
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if dev=="cuda" else torch.float32)
pipe = pipe.to(dev)
return pipe




def generate(prompt, steps, scale, height, width, device):
p = load_pipe(device)
image = p(prompt, num_inference_steps=steps, guidance_scale=scale, height=height, width=width).images[0]
return image




def main():
with gr.Blocks() as demo:
gr.Markdown("# Imaginative Image Generation")
with gr.Row():
prompt = gr.Textbox(label="Prompt", value="a whimsical futuristic treehouse floating on clouds, detailed, vibrant")
with gr.Column():
steps = gr.Slider(1, 50, value=30, label="Inference steps")
scale = gr.Slider(1.0, 12.0, value=7.5, label="Guidance scale")
height = gr.Slider(256, 1024, value=512, step=64, label="Height")
width = gr.Slider(256, 1024, value=512, step=64, label="Width")
device = gr.Radio(choices=["auto", "cuda", "mps", "cpu"], value="auto", label="Device")
out = gr.Image(label="Generated Image")
btn = gr.Button("Generate")
btn.click(fn=generate, inputs=[prompt, steps, scale, height, width, device], outputs=out)


demo.launch()


if __name__ == '__main__':
main()
