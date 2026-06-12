"""Run OmniText training-free text repositioning over OmniText-Bench.

Extracted from the original OmniText_Repositioning notebook (kept verbatim under
notebooks/legacy/). Hyperparameters come from a YAML config (configs/repositioning.yaml);
the attention processor is kept inline because it reads per-step state through the
module namespace, exactly as in the notebook.

Note: this stage consumes the text-removed image produced by run_removal.py
(pass its output directory via ``--removal-output-dir``).

Example:
    python scripts/run_repositioning.py --config configs/repositioning.yaml \
        --dataset-root OmniText-Bench --output-dir experiments/repositioning
"""
import argparse
import os

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/repositioning.yaml", help="Path to the YAML config.")
    p.add_argument("--dataset-root", default="OmniText-Bench", help="Path to the OmniText-Bench directory.")
    p.add_argument("--output-dir", default="experiments/repositioning", help="Where to write per-image outputs.")
    p.add_argument("--removal-output-dir", default="experiments/removal",
                   help="Directory of removal.png outputs (run_removal.py first).")
    p.add_argument("--gpu", default=None, help="CUDA device index (overrides config).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images (debug).")
    return p.parse_args()


args = parse_args()
with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_gpu = args.gpu if args.gpu is not None else str(cfg.get("gpu", 0))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)

import json
import numpy as np
from PIL import Image
from typing import Optional
from tqdm.auto import tqdm
from einops import rearrange
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from diffusers.models.attention_processor import Attention

from omnitext.model.util import load_model
from omnitext.util.ptp import GridAttentionStore
from omnitext.util.util import seed_everything
from omnitext.pipeline.pipeline_v0 import (
    FreeTextStyleTransferPipeline,
    FreeTextStyleTransferPipelineOutput,
)
from omnitext.char_mask import generate_char_masks
from omnitext import data as omnidata

# --- per-iteration state shared with OurAttentionProcessor (module namespace) ---
unet = None
idx_ann = None
target_text = None
source_text_list = []
target_text_list = []
first_text_boost_mask_dict = {}
second_text_boost_mask_dict = {}


class OurAttentionProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        attn_store: GridAttentionStore,
        layer_name: str,
        modulation_type: str,
        is_modulate: bool = False,
        is_save: bool = False,
    ):
        self.attn_store = attn_store
        self.place_in_unet = layer_name.split('_')[0]

        self.layer_name = layer_name
        
        self.modulation_type = modulation_type
        self.counter = 0
        self.is_modulate = is_modulate
        self.is_save = is_save

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        idx_instance: int = -1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_self_attn = True if encoder_hidden_states is None else False

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)  # B H*W C*HEAD
        value = attn.to_v(encoder_hidden_states)  # B H*W C*HEAD

        query = attn.head_to_batch_dim(query)  # B*HEAD H*W C
        key = attn.head_to_batch_dim(key)  # B*HEAD H*W C
        value = attn.head_to_batch_dim(value)  # B*HEAD H*W C

        # OPERATION #
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        batch_size, spatial_size, _ = baddbmm_input.size()

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        if self.is_save:
            BS, spatial_size, _ = attention_scores.size()
            BS = (BS // attn.heads) - 1
            BS = BS * attn.heads

            new_attention_probs = attention_probs.clone()
            if spatial_size == 512:
                h, w = 16, 32
            elif spatial_size == 2048:
                h, w = 32, 64
            
            if is_self_attn:
                first_attn_mask = first_text_boost_mask_dict[idx_ann][spatial_size]
                second_attn_mask = second_text_boost_mask_dict[idx_ann][spatial_size]
                text_mask = first_attn_mask
                first_attn_mask = rearrange(first_attn_mask, "(h w) -> h w", h=h, w=w)
                # print("TEXT MASK")
                # plt.imshow(first_attn_mask.detach().cpu().numpy())
                # plt.show()
                char_masks = generate_char_masks(first_attn_mask, len(source_text_list[idx_ann]), len(target_text_list[idx_ann]))

                masked_tensor = new_attention_probs[BS:, :, :]
                if self.is_modulate:
                    n_text = len(target_text)
                    for idx_char in range(n_text):
                        # print("CHAR MASK", idx_char)
                        # plt.imshow(char_masks[idx_char].detach().cpu().numpy())
                        # plt.show()
                        char_mask = rearrange(char_masks[idx_char], "h w -> (h w)", h=h, w=w)
                        # masked_tensor_inside_text_mask = masked_tensor[:, text_mask == 1, :].mean(dim=1, keepdim=True)
                        # masked_tensor[:, char_mask == 1, :] = masked_tensor_inside_text_mask
    
                        # 6 START
                        B, HW, _ = masked_tensor.shape
                        device = masked_tensor.device

                        # A
                        if self.layer_name in ['up_1_2_self']: # 'up_1_2_self', 
                            masked_tensor[:, char_mask == 1, :] = 0.0
                    
                            # Get the indices of the active region
                            char_indices = torch.where(char_mask == 1)[0]  # [N]
                            N = char_indices.numel()

                            alpha = 1.0
                            if N > 0:
                                identity = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) * alpha  # [B, N, N]
                                masked_tensor[:, char_indices.unsqueeze(0), char_indices.unsqueeze(1)] = identity
                                
                            inside_char = masked_tensor[:, char_mask == 1, :]
                            inside_char[:, :, second_attn_mask == 1] = (1 - alpha) / (second_attn_mask == 1).sum()
                            masked_tensor[:, char_mask == 1, :] = inside_char
                        # print(masked_tensor[:, char_mask == 1, :][:, :, char_mask == 1].size(), masked_tensor[:, char_mask == 1, :][:, :, char_mask == 1].sum())
                        # A END
            
                        # print("AFTER", self.layer_name, masked_tensor.sum(dim=-1).min(), masked_tensor.sum(dim=-1).max(), BS)
                new_attention_probs[BS:, :, :] = masked_tensor
                
                attention_probs = new_attention_probs
                masked_tensor = attention_probs[BS:, :, :]

                if self.layer_name in ['up_1_0_self', 'up_1_1_self']: # '', 
                    self.attn_store(masked_tensor, is_cross=not is_self_attn, place_in_unet=self.place_in_unet)
            else:
                self.attn_store(attention_probs[BS:, :, :], is_cross=not is_self_attn, place_in_unet=self.place_in_unet)
                # print("SAVE", self.attn_store.cur_att_layer, self.attn_store.curr_step_index)


        hidden_states = torch.bmm(attention_probs, value)
        
        del attention_scores
        del attention_probs
        torch.cuda.empty_cache()
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_self_attn_processor(attn_store: GridAttentionStore, modulation_type: str = "boosting", is_modulate: bool = False, is_save: bool = True):
    unet.mid_block.attentions[0].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name="mid_0_0_self", modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    # Not good if we modulate this
    for block_idx in range(0, 1):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))

    # only enable this
    for block_idx in range(1, 2):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    # Not good if we modulate this
    for block_idx in range(2, 3):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))
            
    # for block_idx in range(1, 2):
    #     for layer_idx in range(0, 3):
    #         unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))

    # # only enable this, likely [1, 2] and [2, 0] are the best modulator
    # for block_idx in range(2, 3):
    #     for layer_idx in range(0, 3):
    #         unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=False))

    for block_idx in range(1, 2):
        for layer_idx in range(0, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    for block_idx in range(1, 2):
        for layer_idx in range(0, 3): # 0, 3 default
            # print("Apply Self", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    # unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    for block_idx in range(2, 3):
        for layer_idx in range(0, 1):
            # print("Apply Self", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    for block_idx in range(2, 3):
        for layer_idx in range(1, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))
    # unet.up_blocks[2].attentions[0].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    
    for block_idx in range(3, 4):
        for layer_idx in range(0, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=False, is_save=False))


def set_cross_attn_processor(attn_store: GridAttentionStore, modulation_type: str = "boosting", is_modulate: bool = False, is_save: bool = True):
    unet.mid_block.attentions[0].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"mid_0_0_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    # Not good if we modulate this
    for block_idx in range(0, 1):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))

    # only enable this
    for block_idx in range(1, 2):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    # Not good if we modulate this
    for block_idx in range(2, 3):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))

    for block_idx in range(1, 2):
        for layer_idx in range(0, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))
    # for block_idx in range(1, 2):
    #     for layer_idx in [2]:
    #         # print("Apply Cross", block_idx, layer_idx)
    #         unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    # # unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    for block_idx in range(1, 2):
        for layer_idx in range(2, 3):
            # print("Apply Cross", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    # unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    for block_idx in range(2, 3):
        for layer_idx in range(0, 1):
            # print("Apply Cross", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    for block_idx in range(2, 3):
        for layer_idx in range(1, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))
    # unet.up_blocks[2].attentions[0].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    
    for block_idx in range(3, 4):
        for layer_idx in range(0, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=False, is_save=False))

def main():
    global unet, idx_ann, target_text, source_text_list, target_text_list
    global first_text_boost_mask_dict, second_text_boost_mask_dict

    device = "cuda"
    dtype = torch.float32 if str(cfg.get("dtype", "float32")) == "float32" else torch.float16
    seed_everything(seed=cfg.get("model_seed", 42))

    tokenizer, text_encoder, vae, unet, scheduler = load_model(dtype=dtype, device=device)

    valid_res = cfg.get("valid_res", [512, 2048])
    attn_store = GridAttentionStore(valid_res=valid_res)
    self_attention_list = []

    pipe = FreeTextStyleTransferPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipe.to(device)

    seed = cfg.get("seed", 8)
    generator = torch.Generator("cuda").manual_seed(seed)
    num_inference_steps = cfg.get("num_inference_steps", 20)
    guidance_scale = cfg.get("guidance_scale", 3)
    strength = cfg.get("strength", 1.0)

    labels = omnidata.load_labels(args.dataset_root)
    if args.limit is not None:
        labels = {k: labels[k] for k in list(labels)[: args.limit]}

    out_dir = args.output_dir
    ref = "ref1"

    pipe._guidance_scale = guidance_scale
    batch_size = 1
    device = pipe._execution_device
    dtype = pipe.dtype

    new_anns = {}
    for key in tqdm(labels):
        with torch.no_grad():
            anns = labels[key]['application']['repositioning']
            out_subdir = os.path.join(out_dir, key)
            Path(out_subdir).mkdir(parents=True, exist_ok=True)
        
            ref_image_path = str(omnidata.input_image_path(args.dataset_root, key))
            ref_image = Image.open(ref_image_path).convert("RGB")
    
            inp_image_path = str(omnidata.removal_output_path(args.removal_output_dir, key))
            inp_image = Image.open(inp_image_path).convert("RGB")
    
            image_width, image_height = inp_image.size
            image_size = (image_height, image_width)
        
            # 1. prepare image grid
            width, height = inp_image.size
            ref_image_array = np.array(ref_image)
            inp_image_array = np.array(inp_image)
    
            grid_size = 2
            image_grid_array = np.zeros((height, width * grid_size, 3), dtype=np.uint8)
            image_grid_array[:, :512, :] = inp_image_array
            image_grid_array[:, 512:, :] = ref_image_array
            image_grid = Image.fromarray(image_grid_array)
            # display(image_grid)
    
            # plt.imshow(image_grid)
            # plt.show()
    
            # 4. prepare latents
            num_channels_latents = pipe.vae.config.latent_channels
            num_channels_unet = pipe.unet.config.in_channels
            return_image_latents = num_channels_unet == 4
            grid_width, grid_height = image_grid.size
    
            noise = pipe.prepare_latents(
                num_channels_latents,
                grid_height,
                grid_width,
                device,
                dtype,
                generator=generator,
            )
            latents = pipe.prepare_latents(
                num_channels_latents,
                grid_height,
                grid_width,
                device,
                dtype,
                generator=generator,
            )
            image_tensor = pipe.image_processor.preprocess(image_grid, height=grid_height, width=grid_width).to(device=device, dtype=torch.float32)
            image_latent = pipe._encode_vae_image(image_tensor, generator)
        
            new_latents = pipe.scheduler.add_noise(
                original_samples=image_latent, 
                noise=noise, 
                timesteps=torch.tensor([999]).to(image_latent.device)
            )
            latents = new_latents
        
            # 5. prepare timesteps for sampling
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipe.scheduler.timesteps
             
            # 6. prepare editing conditions
            editing_feature_mask_list = []
            editing_masked_feature_list = []
            editing_encoder_hidden_states_list = []
            image_latent_mask_list = []
            first_text_boost_mask_dict = {}
            second_text_boost_mask_dict = {}
    
            target_text_list = []
            source_text_list = []
            for idx_ann in range(1):
                source_text = anns['text']
                target_text = anns['text']
                if True:
                    source_text_list.append(target_text)
                
                    target_text_list.append(target_text)
                    first_text_boost_mask_dict[idx_ann] = {}
                    second_text_boost_mask_dict[idx_ann] = {}
        
                
                    ref_mask_polygon = anns['removal_polygon']
                    target_mask_polygon = anns['target_polygon']
                
                    # print(mask_polygon)
                    prompt = target_text

                    target_mask_polygon = np.array(target_mask_polygon).reshape(-1, 2)
                    reference_mask_polygon = np.array(ref_mask_polygon).reshape(-1, 2)
                            
                    target_mask_polygon = target_mask_polygon.astype(int)
                    target_mask_polygon = target_mask_polygon.flatten().tolist()
                
                    reference_mask_polygon = reference_mask_polygon.astype(int)
                    reference_mask_polygon = reference_mask_polygon.flatten().tolist()

                    
                    with torch.no_grad():
                        # 6.1. prepare mask
                        grid_width, grid_height = image_grid.size
                        grid_size = (grid_height, grid_width)
        
                        # for editing
                        editing_grid_mask, editing_grid_mask_bbox = pipe.prepare_curved_image_mask(
                            target_mask_polygon,
                            grid_size,
                            device=device,
                            dtype=dtype,
                        )
                        # plt.imshow(editing_grid_mask.squeeze().cpu().numpy())
                        # plt.show()
                    
                        # 6.2. Encode edit prompt
                        editing_prompt_embeds_cond, editing_prompt_embeds_uncond, text_input_ids = pipe.encode_prompt(
                            prompt=prompt,
                            mask_bbox=editing_grid_mask_bbox, # matters for editing, does not matter for removal
                            device=device,
                            negative_prompt=None,
                        )
                        editing_encoder_hidden_states = torch.cat([editing_prompt_embeds_uncond, editing_prompt_embeds_cond])

                        reference_grid_mask, reference_grid_mask_bbox = pipe.prepare_curved_image_mask(
                            reference_mask_polygon,
                            grid_size,
                            device=device,
                            dtype=dtype
                        )
                    
                        editing_grid_masked_latent, editing_grid_latent_mask = pipe.prepare_inpaint_input(
                            image=image_grid,
                            image_mask=editing_grid_mask,
                            device=device,
                            dtype=dtype,
                            generator=generator,
                        )
                        _, reference_grid_latent_mask = pipe.prepare_inpaint_input(
                            image=image_grid,
                            image_mask=reference_grid_mask,
                            device=device,
                            dtype=dtype,
                            generator=generator
                        )
                    
                        editing_masked_latent = editing_grid_masked_latent
                        editing_latent_mask = editing_grid_latent_mask
                        reference_latent_mask = reference_grid_latent_mask
        
                        latents_channel = latents.size(1)
                        masked_latent_channel = editing_masked_latent.size(1)
                        latent_mask_channel = editing_latent_mask.size(1)
                        assert latents_channel + masked_latent_channel + latent_mask_channel == 9
                        assert latents.size(2) == editing_masked_latent.size(2) == editing_latent_mask.size(2)
                        assert latents.size(3) == editing_masked_latent.size(3) == editing_latent_mask.size(3)
        
                        editing_encoder_hidden_states_list.append(editing_encoder_hidden_states)
                        editing_masked_feature_list.append(torch.cat([editing_masked_latent] * 2))
                        editing_feature_mask_list.append(torch.cat([editing_latent_mask] * 2))

                        first_self_attn_mask = editing_latent_mask.clone()
                        second_self_attn_mask = reference_latent_mask.clone()
                        second_self_attn_mask[:, :, :, 64:] = second_self_attn_mask[:, :, :, :64]
                        second_self_attn_mask[:, :, :, :64] = 0
                    
                        # plt.imshow(first_self_attn_mask.squeeze().cpu().numpy())
                        # plt.show()
                        # plt.imshow(second_self_attn_mask.squeeze().cpu().numpy())
                        # plt.show()

                        # print("HEHE")
                        # plt.imshow(first_self_attn_mask.squeeze().cpu().numpy())
                        # plt.show()
                        # plt.imshow(second_self_attn_mask.squeeze().cpu().numpy())
                        # plt.show()
                    
                        _, _, H, W = first_self_attn_mask.size()
                        for scale_factor in [1, 0.5, 0.25, 0.125]:
                            h, w = int(scale_factor * H), int(scale_factor * W)
                            resized_first_text_mask = F.interpolate(first_self_attn_mask, size=(h, w), mode="bilinear")
                            resized_first_text_mask[resized_first_text_mask > 0] = 1
                            resized_first_text_mask[resized_first_text_mask <= 0] = 0
                            resized_first_text_mask = resized_first_text_mask[0, 0] # ENABLE OR DISABLE THIS
                            resized_second_text_mask = F.interpolate(second_self_attn_mask, size=(h, w), mode="bilinear")
                            resized_second_text_mask = resized_second_text_mask[0, 0]
                            resized_second_text_mask[resized_second_text_mask > 0] = 1
                            resized_second_text_mask[resized_second_text_mask <= 0] = 0
    
                            # plt.imshow(resized_first_text_mask.cpu().numpy())
                            # if idx_ann == 4 and scale_factor == 1:
                            #     plt.savefig("boost_first_mask.png")
                            # plt.show()
                            # plt.imshow(resized_second_text_mask.cpu().numpy())
                            # if idx_ann == 4 and scale_factor == 1:
                            #     plt.savefig("boost_second_mask.png")
                            # plt.show()
    
                            spatial_size = h * w 
                            first_text_boost_mask_dict[idx_ann][spatial_size] = resized_first_text_mask.view(-1)
                            second_text_boost_mask_dict[idx_ann][spatial_size] = resized_second_text_mask.view(-1)
    
    
    
            # 8. Sampling
            attn_store = GridAttentionStore(valid_res=valid_res)
            attn_store.num_att_layers = 4 #
            adaptive_scale_con = 5 # 2 can also be used
            adaptive_scale_self_style = 10
            with torch.no_grad():
                with torch.enable_grad():
                    for i, t in tqdm(enumerate(timesteps), leave=False):
                        optimized_latents = latents.clone().detach().requires_grad_(True)
                        optimizer = Adam([optimized_latents], lr=1e-2, eps=1e-3) # 5e-2 # SGD doesn't work, 
    
                    
                        iteration = 0
                        loss = torch.tensor(10000)
    
                        if i == 0 or i == 4 or i == 8: # i>=3 and i < 8
                            while loss.abs().item() > 0.01 and iteration < 20:
                                optimized_latents.requires_grad = True
    
                                loss = 0.0
                                for idx_ann, (editing_encoder_hidden_states, editing_feature_mask, editing_masked_feature) in enumerate(zip(
                                    editing_encoder_hidden_states_list, editing_feature_mask_list, editing_masked_feature_list
                                )):
                                    set_self_attn_processor(attn_store=attn_store, modulation_type="boosting", is_modulate=True, is_save=True)
                                    set_cross_attn_processor(attn_store=attn_store, is_modulate=False, is_save=True)
                                    noise_pred = pipe.unet(
                                        sample=optimized_latents, 
                                        timestep=t,
                                        encoder_hidden_states=editing_encoder_hidden_states[1:], 
                                        feature_mask=editing_feature_mask[1:], 
                                        masked_feature=editing_masked_feature[1:]
                                    ).sample # b, 4, 64, 64
                                    del noise_pred
                                    torch.cuda.empty_cache()
    
                                    # Cross Attention
                                    n_text = len(target_text_list[idx_ann])
                                    cross_attention_maps = attn_store.aggregate_attention(from_where=["up"], cutoff_index=7+len(target_text_list[idx_ann]), is_cross=True) # H W C
    
                                    h = 32
                                    w = 64
                                    first_attn_mask = rearrange(first_text_boost_mask_dict[idx_ann][2048], "(h w) -> h w", h=h, w=w)[:, :32].contiguous()
                                    text_mask = first_attn_mask
                                    char_masks = generate_char_masks(text_mask, len(source_text_list[idx_ann]), len(target_text_list[idx_ann]))
    
                                    constraint_loss = 0
                                    for idx_char in range(len(target_text_list[idx_ann])):
                                        idx_char_in_probs = 7 + idx_char
                                        char_mask = char_masks[idx_char]
                                    
                                        attn_map = cross_attention_maps[:, :, idx_char_in_probs]
                                        attn_map = attn_map * text_mask
    
                                        # plt.imshow(attn_map.detach().cpu().numpy())
                                        # plt.show()
                                        # plt.imshow(char_mask.detach().cpu().numpy())
                                        # plt.show()
                                        attn_map = attn_map.view(-1)
                                        flatten_text_mask = text_mask.view(-1)
                                        char_mask = char_mask.view(-1)
        
                                        eps = 1e-8
                                        alpha = 0.9
                                        gamma = 2.0
                                        masked_pred = attn_map[flatten_text_mask == 1]
                                        masked_target = char_mask[flatten_text_mask == 1].float()
                                        inputs = masked_pred
                                        targets = masked_target
                                        inputs = inputs.clamp(min=eps, max=1.0 - eps)  # for numerical stability
                                        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
                                        pt = torch.where(targets == 1, inputs, 1 - inputs)
                                        focal_weight = alpha * (1 - pt) ** gamma
                                        constraint_loss += (focal_weight * bce_loss).mean()
                                    
    
                                    constraint_loss /= len(target_text_list[idx_ann])
    
                                    if adaptive_scale_con is None:
                                        adaptive_scale_con = 15 / constraint_loss.detach()
                                        # print(adaptive_scale_con)
    
                                    constraint_loss *= adaptive_scale_con
    

                                    self_attention_loss = 0
                                    self_attention_map_list = attn_store.aggregate_attention(from_where=["up"], cutoff_index=None, is_cross=False)
                                    for self_attn_map in self_attention_map_list:
                                        # print(len(self_attention_map_list))
                                        bs, _, spatial_size = self_attn_map.size() # this self attention map is already inside the text (first_attn_mask == 1)
                                    
                                        if spatial_size == 512:
                                            h, w = 16, 32
                                        else:
                                            h, w = 32, 64
    
                                        first_attn_mask = first_text_boost_mask_dict[idx_ann][spatial_size]
                                        text_mask = rearrange(first_attn_mask, "(h w) -> h w", h=h, w=w)
                                        char_masks = generate_char_masks(text_mask, len(source_text_list[idx_ann]), len(target_text_list[idx_ann]))
    
                                        second_attn_mask = second_text_boost_mask_dict[idx_ann][spatial_size]
    
                                        pred_distribution = self_attn_map[:, first_attn_mask == 1, :]
                                        pred_distribution = pred_distribution.mean(dim=0)
                                    
                                        target_distribution = second_attn_mask.clone()
                                        target_distribution = target_distribution / (target_distribution.sum(-1, keepdim=True) + 1e-8)
    
                                        # print(pred_distribution.size(), target_distribution.size())
    
                                        kl_loss = F.kl_div(pred_distribution.log(), target_distribution.expand_as(pred_distribution), reduction='batchmean')
                                        self_attention_loss += kl_loss
                                        # G End

                                    self_attention_loss /= len(self_attention_map_list)
    
                                    if adaptive_scale_self_style is None:
                                        adaptive_scale_self_style = 5 / self_attention_loss.detach()
                                    self_attention_loss *= adaptive_scale_self_style
    
                                    # print(adaptive_scale_self_acc, constraint_loss, self_attention_loss)
                                    loss += constraint_loss + self_attention_loss
                                    # self_attention_loss + self_loss works well 
                                    # print("AFTER", loss, self_attention_loss, self_char_loss, constraint_loss)
    
                                loss /= len(editing_encoder_hidden_states_list)
                                if loss != 0:
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                iteration += 1
                                torch.cuda.empty_cache()
    
                        set_self_attn_processor(attn_store=attn_store, modulation_type="boosting", is_modulate=False, is_save=False)
                        set_cross_attn_processor(attn_store=attn_store, is_modulate=False, is_save=False)
                        optimized_latents = optimized_latents.detach().requires_grad_(False)
                        with torch.no_grad():
                            # latents.requires_grad = False
                            latents = optimized_latents # freq_mix_2d(optimized_latents, optimized_latents_text, LPF) # Style Optimization
                            latent_model_input = torch.cat([latents, latents])
                            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
                            noisy_residual_editing_list = []
                            for idx_ann, (editing_encoder_hidden_states, editing_feature_mask, editing_masked_feature) in enumerate(zip(
                                editing_encoder_hidden_states_list, editing_feature_mask_list, editing_masked_feature_list
                            )):
                                noise_pred = pipe.unet(
                                    sample=latent_model_input, 
                                    timestep=t, 
                                    encoder_hidden_states=editing_encoder_hidden_states, 
                                    feature_mask=editing_feature_mask, 
                                    masked_feature=editing_masked_feature
                                ).sample # b, 4, 64, 64
                                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                                noisy_residual_editing = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64
    
                            
                            
                                noisy_residual_editing_list.append(noisy_residual_editing) #editing_feature_mask[:1, :, :, :]
                            noisy_residual_editing = torch.sum(torch.stack(noisy_residual_editing_list), dim=0)
        
                            noisy_residual = noisy_residual_editing
                        
                            prev_latents = pipe.scheduler.step(noisy_residual, t, latents).prev_sample
                            # pred_original_sample = pipe.scheduler.step(noisy_residual, t, latents).pred_original_sample
                
                            # pred_original_sample_view = pipe.vae.decode(
                            #     pred_original_sample / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
                            # )[0]
                            # pred_original_sample_view = pipe.image_processor.postprocess(pred_original_sample_view, output_type="pil", do_denormalize=[True] * pred_original_sample.shape[0])
                            # pred_original_sample_view = pred_original_sample_view[0]
                            # pred_original_sample_view = np.array(pred_original_sample_view)
                            # plt.imshow(pred_original_sample_view)
                            # plt.show()
                            latents = prev_latents
    
            
                output_image = pipe.vae.decode(
                    latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
                )[0]
                output_image = pipe.image_processor.postprocess(output_image, output_type="pil", do_denormalize=[True] * output_image.shape[0])
    
                out_path = os.path.join(out_subdir, f"{idx_ann:02d}.png")
    
                output_image = output_image[0]
                output_image = np.array(output_image)
                output_image = output_image[:, :512, :]
                output_image = Image.fromarray(output_image)
                output_image.save(out_path)
    
                # plt.imshow(output_image)
                # plt.show()


if __name__ == "__main__":
    main()
