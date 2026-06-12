import string
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from .custom_model_v2 import MyUNet2DConditionModel

# Maintained community mirror of the original ``runwayml/stable-diffusion-v1-5``,
# which was removed from the Hugging Face Hub in 2024.
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
TEXTDIFFUSER2_REPO = "JingyeChen22/textdiffuser2-full-ft-inpainting"


def load_model(dtype=torch.float16, device="cuda"):
    print("Loading the model")

    tokenizer = CLIPTokenizer.from_pretrained(
        SD15_REPO, subfolder="tokenizer"
    )
    
    alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
    '''alphabet
    0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
    '''
    
    #### additional tokens are introduced, including coordinate tokens and character tokens
    print('***************')
    print("Number of tokens in original tokenizer", len(tokenizer))
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) 
    print("Number of tokens after expansion of tokenizer", len(tokenizer))
    print('***************')

    text_encoder = CLIPTextModel.from_pretrained(
        TEXTDIFFUSER2_REPO, subfolder="text_encoder"
    ).to(dtype=dtype)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Loaded from the Hugging Face Hub (cached by scripts/download_weights.py);
    # no manual ``git clone`` of the VAE is required.
    vae = AutoencoderKL.from_pretrained(VAE_REPO).to(dtype=dtype)

    # The inpainting UNet config is patched to in_channels=9 by
    # scripts/download_weights.py before loading.
    unet = MyUNet2DConditionModel.from_pretrained(
        TEXTDIFFUSER2_REPO, subfolder="unet"
    ).to(dtype=dtype)

    scheduler = DDPMScheduler.from_pretrained(SD15_REPO, subfolder="scheduler")

    return tokenizer, text_encoder, vae, unet, scheduler
