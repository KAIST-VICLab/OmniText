import numpy as np
import random
import torch
import torchvision.transforms as T

from diffusers import AutoencoderKL
from PIL import Image

def seed_everything(seed=7):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU, if available)
    torch.cuda.manual_seed_all(seed)  # All GPUs

def to_tensor(image: Image):
    image = np.array(image)
    image = image / 127.5 - 1
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    return image

def encode_images(images: torch.Tensor, vae: AutoencoderKL):
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = vae.config.scaling_factor * latents
    return latents

def decode_latents(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / vae.config.scaling_factor * latents 
    image = vae.decode(latents, return_dict=False)[0] 
    return image

def to_image(images: torch.Tensor):
    image = images[0]
    image = (image + 1) / 2
    to_pil = T.ToPILImage()
    image = to_pil(image)
    return image