import torch
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe.to("cpu")

# Disable safety checker to avoid TensorFlow error (optional)
pipe.safety_checker = None

# Shortened prompt to fit within 77 tokens
prompt = "A Lion in the jungle with a sunset in the background."

# Generate the image
image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=1.5,
    num_inference_steps=50,  # Reduced to hit ~5 minutes
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]

# Save the image
image.save("sd_demo.png")

# Optional: Upscale
from PIL import Image
image = Image.open("sd_demo.png").resize((1024, 1024), Image.LANCZOS)
image.save("sd_demo_upscaled.png")