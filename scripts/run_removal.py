"""Run OmniText training-free text removal over OmniText-Bench.

Removal is the first stage of the OmniText pipeline: editing, repositioning and
rescaling consume its ``removal.png`` output. Hyperparameters come from a YAML config
(see configs/removal.yaml); the attention processor is kept inline (it reads per-step
state through the module namespace, exactly as in the original notebook).

Example:
    python scripts/run_removal.py --config configs/removal.yaml \
        --dataset-root OmniText-Bench --output-dir experiments/removal
"""
import argparse
import os

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/removal.yaml", help="Path to the YAML config.")
    p.add_argument("--dataset-root", default="OmniText-Bench", help="Path to the OmniText-Bench directory.")
    p.add_argument("--output-dir", default="experiments/removal", help="Where to write per-image outputs.")
    p.add_argument("--gpu", default=None, help="CUDA device index (overrides config).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images (debug).")
    return p.parse_args()


args = parse_args()
with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

# CUDA_VISIBLE_DEVICES must be set before importing torch.
_gpu = args.gpu if args.gpu is not None else str(cfg.get("gpu", 0))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)

import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Optional
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from omnitext.model.util import load_model
from omnitext.util.ptp import GridAttentionStore
from omnitext.util.util import seed_everything
from omnitext.pipeline.pipeline_v0_removal import (
    FreeTextStyleTransferPipeline,
    FreeTextStyleTransferPipelineOutput,
)
from omnitext import data as omnidata

# --- per-iteration state shared with OurAttentionProcessor (module namespace) ---
unet = None
alpha = None
text_removal_mask_dict = {}
first_text_boost_mask_dict = {}
second_text_boost_mask_dict = {}
idx_ann = None


class OurAttentionProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        attn_store: GridAttentionStore,
        layer_name: str,
        modulation_type: str,
        alpha: torch.tensor = 0,
        is_modulate: bool = False,
        is_save: bool = False,
        suppression_scope: str = "all"
    ):
        self.attn_store = attn_store
        self.place_in_unet = layer_name.split('_')[0]

        self.layer_name = layer_name
        # self.alpha = alpha
        
        self.modulation_type = modulation_type
        self.counter = 0
        self.is_modulate = is_modulate
        self.is_save = is_save

        self.suppression_scope = suppression_scope

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

        if self.is_modulate:
            if is_self_attn:
                assert self.modulation_type in ["boosting", "removal"]
                if self.modulation_type == "boosting":
                    BS, spatial_size, _ = attention_scores.size()
                    BS //= 2 # only apply to conditional
                    first_attn_mask = first_text_boost_mask_dict[idx_ann][spatial_size]
                    second_attn_mask = second_text_boost_mask_dict[idx_ann][spatial_size]

                    attention_probs = attention_scores.softmax(dim=-1)

                    masked_tensor = attention_probs[BS:, first_attn_mask == 1]

                    # B. Proportional Redistribution Based on Existing Values
                    transfer_ratio = 0.5
                    unselected_mass = masked_tensor[:, :, second_attn_mask == 0].sum()
                
                    selected_values = masked_tensor[:, :, second_attn_mask == 1]
                    selected_distribution = selected_values / selected_values.sum()
                
                    # Transfer mass proportionally
                    transfer_amount = transfer_ratio * unselected_mass
                    masked_tensor[:, :, second_attn_mask == 1] += transfer_amount * selected_distribution
                
                    # Reduce mass from unselected regions proportionally
                    masked_tensor[:, :, second_attn_mask == 0] *= (1 - transfer_ratio)
                    
                    attention_probs[BS:, first_attn_mask == 1] = masked_tensor
                    attention_probs /= attention_probs.sum(dim=-1, keepdim=True) + 1e-6
                    
                    attention_probs = attention_probs.to(dtype)
                else:
                    BS, spatial_size, _ = attention_scores.size()

                    # first_attn_mask = first_text_removal_mask_dict[spatial_size] # .unsqueeze(0).repeat(BS, 1)
                    # second_attn_mask = second_text_removal_mask_dict[spatial_size].clone()

                    # sreg = 1.0 # CAN BE HIGHER
                    # min_value = attention_scores[:].min(-1)[0].unsqueeze(-1) # B x HW x 1

                    # masked_tensor = attention_scores[:, first_attn_mask == 1]                    
                    # masked_tensor[:, :, second_attn_mask == 1] = min_value[:, first_attn_mask == 1, :]
                    # attention_probs = attention_scores.softmax(dim=-1)
                    # attention_probs = attention_probs.to(dtype)

                    ###
                    attn_mask = text_removal_mask_dict[spatial_size].clone()
                    # masked_tensor = attention_scores[:, attn_mask == 1]
                    # min_value = masked_tensor[:].min(-1)[0].unsqueeze(-1) # B x HW x 1
                    
                    # if self.suppression_scope == 'all':
                    #     masked_tensor[:, :, :] = min_value
                    # else:
                    #     masked_tensor[:, :, attn_mask == 1] = min_value
                        
                    # attention_scores[:, attn_mask == 1] = masked_tensor
                    attention_probs = attention_scores.softmax(dim=-1)

                    # flipping
                    masked_tensor = attention_probs[:, attn_mask == 1]                    
                    max_value = masked_tensor[:].max(-1)[0].unsqueeze(-1) # B x HW x 1
                    min_value = masked_tensor[:].min(-1)[0].unsqueeze(-1) # B x HW x 1
                    flipped = max_value + min_value - masked_tensor
                    flipped = flipped.softmax(dim=-1) # flipped / flipped.sum()
                    attention_probs[:, attn_mask == 1] = flipped
                    
                    attention_probs = attention_probs.to(dtype)
                    ###

                    # if is_self_attn:
                    #     self_attention_list.append(attention_probs.detach())

                    # if self.is_save:
                    #     self.attn_store(attention_probs, is_cross=not is_self_attn, place_in_unet=self.place_in_unet)
            else:
                # cross attention
                BS, spatial_size, _ = attention_scores.size()
                BS = 0 # only apply to conditional
                
                # min_value = attention_scores[BS:].min(-1)[0].unsqueeze(-1)
                # max_value = attention_scores[BS:].max(-1)[0].unsqueeze(-1)

                attention_probs = attention_scores.softmax(dim=-1)
                attention_probs = attention_probs.to(dtype)

                # HOW TO OPTIMIZE THIS?
                new_attention_probs = attention_probs.clone()
                
                attn_mask = text_removal_mask_dict[spatial_size].clone()
                
                transfer = alpha * new_attention_probs[BS:, attn_mask==1, [0]]
                new_attention_probs[BS:, attn_mask==1, [0]] -= transfer
                new_attention_probs[BS:, attn_mask==1, [1]] += transfer

                transfer = alpha * new_attention_probs[BS:, attn_mask==0, [1]]
                new_attention_probs[BS:, attn_mask==0, [0]] += transfer
                new_attention_probs[BS:, attn_mask==0, [1]] -= transfer
                
                new_attention_probs[BS:, attn_mask==1, [1]] = new_attention_probs[BS:, attn_mask==1, [1]] + new_attention_probs[BS:, attn_mask==1, 2:].sum(-1)
                new_attention_probs[BS:, attn_mask==0, [0]] = new_attention_probs[BS:, attn_mask==0, [0]] + new_attention_probs[BS:, attn_mask==0, 2:].sum(-1)
                # new_attention_probs[BS:, :, 2:] = 0

                new_attention_probs[BS:, attn_mask==1, 2:] = 0
                
                attention_probs = new_attention_probs

            
        else:
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(dtype)
        
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


def set_self_attn_processor(attn_store: GridAttentionStore, modulation_type: str = "boosting", is_modulate: bool = False, is_save: bool = True, suppression_scope: str = "all"):
    # unet.mid_block.attentions[0].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name="mid_0_0_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    
    # Not good if we modulate this
    # for block_idx in range(0, 1):
    #     for layer_idx in range(0, 2):
    #         unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=False))

    # only enable this
    # for block_idx in range(1, 2):
    #     for layer_idx in range(0, 2):
    #         unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    
    # # Not good if we modulate this
    # for block_idx in range(2, 3):
    #     for layer_idx in range(0, 2):
    #         unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
            
    # for block_idx in range(1, 2):
    #     for layer_idx in range(0, 3):
    #         unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    # # only enable this, likely [1, 2] and [2, 0] are the best modulator
    for block_idx in range(2, 3):
        for layer_idx in [1]: # range(0, 2):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save, suppression_scope=suppression_scope))

    # for block_idx in range(3, 4):
    #     for layer_idx in range(0, 3):
    #         unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn1.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_self", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))


def set_cross_attn_processor(attn_store: GridAttentionStore, alpha: float = 0.5, modulation_type: str = "boosting", is_modulate: bool = False, is_save: bool = True, suppression_scope: str = "all"):
    unet.mid_block.attentions[0].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"mid_0_0_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    # Not good if we modulate this
    for block_idx in range(0, 1):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))

    # only enable this
    # 10 error, 11 okay no impact
    for block_idx in range(1, 2):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))

    # Not good if we modulate this
    # 20 okay, 21 okay -> no impact
    for block_idx in range(2, 3):
        for layer_idx in range(0, 2):
            unet.down_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))
    unet.down_blocks[2].attentions[1].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"down_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))
            
    for block_idx in range(1, 2):
        for layer_idx in range(0, 2):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))
    
    for block_idx in range(1, 2):
        for layer_idx in range(2, 3):
            # print("Apply Cross", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    # unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    # only enable this
    for block_idx in range(2, 3):
        for layer_idx in range(0, 1):
            # print("Apply Cross", block_idx, layer_idx)
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))
    
    for block_idx in range(2, 3):
        for layer_idx in range(1, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))
    # unet.up_blocks[2].attentions[0].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", modulation_type=modulation_type, is_modulate=is_modulate, is_save=is_save))

    for block_idx in range(3, 4):
        for layer_idx in range(0, 3):
            unet.up_blocks[block_idx].attentions[layer_idx].transformer_blocks[0].attn2.set_processor(OurAttentionProcessor(attn_store=attn_store, layer_name=f"up_{block_idx}_{layer_idx}_cross", alpha=alpha, modulation_type=modulation_type, is_modulate=False, is_save=False))

def main():
    global unet, alpha, text_removal_mask_dict

    device = "cuda"
    dtype = torch.float16 if str(cfg.get("dtype", "float16")) == "float16" else torch.float32
    seed_everything(seed=cfg.get("model_seed", 42))

    tokenizer, text_encoder, vae, unet, scheduler = load_model(dtype=dtype, device=device)

    valid_res = cfg.get("valid_res", [512, 512])
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

    labels = omnidata.load_labels(args.dataset_root)
    if args.limit is not None:
        labels = {k: labels[k] for k in list(labels)[: args.limit]}

    img_dir = os.path.join(args.dataset_root, "Input")
    out_dir = args.output_dir

    pipe._guidance_scale = guidance_scale
    batch_size = 1
    device = pipe._execution_device
    dtype = pipe.dtype

    new_anns = {}
    for key in tqdm(labels):
        output_stack = []
        with torch.no_grad():
            anns = labels[key]['application']['removal']
    
            out_subdir = os.path.join(out_dir, key)
            Path(out_subdir).mkdir(parents=True, exist_ok=True)
        
            image_path = str(omnidata.input_image_path(args.dataset_root, key))
            image = Image.open(image_path).convert("RGB")
            output_stack.append(image)
            
            width, height = image.size
            image_size = (height, width)
        
            # 1. prepare image grid
            image_array = np.array(image)
        
            grid_size = 1
            image_grid_array = np.zeros((height, width * grid_size, 3), dtype=np.uint8)
            image_grid_array[:, :512, :] = image_array
            image_grid = Image.fromarray(image_grid_array)
        
            # 3. prepare conditions for removal
            # 3.1. prepare removal mask
            image_mask_list = []
        
            if True:
                mask_polygon = anns['polygon']
    
                polygon = np.array(mask_polygon).reshape(-1, 2)
                rect = cv2.minAreaRect(polygon)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.float32).flatten().tolist()
    
                mask_polygon = box
    
                with torch.no_grad():
                    # 3.1. prepare mask
                    grid_width, grid_height = image_grid.size
                    grid_size = (grid_height, grid_width)
    
                    image_grid_mask = pipe.prepare_curved_image_mask(
                        mask_polygon,
                        grid_size,
                        image_size,
                        device=device,
                        dtype=dtype
                    )
                
                    image_mask_list.append(image_grid_mask)
    
            # 3.2. prepare text embeds and others
            edited_image_mask = torch.clamp(torch.sum(torch.stack(image_mask_list), dim=0), 0, 1)
    
            removal_mask_bbox = (0, 0, 0, 0)
            removal_prompt_embeds_cond, removal_prompt_embeds_uncond, removal_input_ids = pipe.encode_prompt(
                prompt="",
                mask_bbox=removal_mask_bbox, # does not matter for removal
                device=device,
                negative_prompt=None,
            )
            removal_encoder_hidden_states = torch.cat([removal_prompt_embeds_uncond, removal_prompt_embeds_cond])
        
            removal_grid_masked_latent, removal_grid_latent_mask = pipe.prepare_inpaint_input(
                image=image_grid,
                image_mask=edited_image_mask,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            removal_masked_feature = torch.cat([removal_grid_masked_latent] * 2)
            removal_feature_mask = torch.cat([removal_grid_latent_mask] * 2)
    
            # image latent
            width, height = image_grid.size
            image_tensor = pipe.image_processor.preprocess(image_grid, height=height, width=width).to(device=device, dtype=dtype)
            image_latent = pipe._encode_vae_image(image_tensor, generator)
            latent_mask = removal_grid_latent_mask
    
            # 3.3. Prepare mask for removal modulation
            self_attn_mask = removal_grid_latent_mask.clone()
        
            _, _, H, W = self_attn_mask.size()
            text_removal_mask_dict = {}
            for scale_factor in [1, 0.5, 0.25, 0.125]:
                h, w = int(scale_factor * H), int(scale_factor * W)
                resized_text_mask = F.interpolate(self_attn_mask, size=(h, w), mode='bilinear')
                resized_text_mask[resized_text_mask < 0] = 0
                resized_text_mask[resized_text_mask > 0] = 1
            
                spatial_size = h * w
                text_removal_mask_dict[spatial_size] = resized_text_mask.view(-1)
    
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
            latents = pipe.scheduler.add_noise(
                original_samples=image_latent,
                noise=noise,
                timesteps=torch.tensor([999]).to(image_latent.device)
            )
            
            # 5. prepare timesteps for sampling
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipe.scheduler.timesteps
    
            attn_store.num_att_layers = 2
            removal_latents = latents.clone()
            with torch.no_grad():
                with torch.enable_grad():
                    attn_store.num_att_layers = 4
                    for i, t in tqdm(enumerate(timesteps), leave=False):
                        if i < 20:
                            alpha = torch.tensor(1.0, device=latents.device, dtype=latents.dtype).requires_grad_(False)
                        # elif i >= 5 and i < 10:
                        #     alpha = torch.tensor(0.5, device=latents.device, dtype=latents.dtype).requires_grad_(False)
                        # elif i >= 8 and i < 12:
                        #     alpha = torch.tensor(0.0, device=latents.device, dtype=latents.dtype).requires_grad_(False)
                        # elif i >= 12 and i < 16:
                        #     alpha = torch.tensor(0.25, device=latents.device, dtype=latents.dtype).requires_grad_(False)
                        # else:
                        #     alpha = torch.tensor(0, device=latents.device, dtype=latents.dtype).requires_grad_(False)
                    
                        removal_latents.requires_grad = False
                        removal_latent_model_input = torch.cat([removal_latents] * 2)
                        removal_latent_model_input = pipe.scheduler.scale_model_input(removal_latent_model_input, t)
                    
                        # removal
                        if i < 10:
                            set_self_attn_processor(attn_store=attn_store, modulation_type="removal", is_modulate=True, is_save=True, suppression_scope="all")
                        else:
                            set_self_attn_processor(attn_store=attn_store, modulation_type="removal", is_modulate=False, is_save=True)
                        set_cross_attn_processor(attn_store=attn_store, modulation_type="removal", alpha=alpha, is_modulate=True, is_save=False)
                        noise_pred = pipe.unet(
                            sample=removal_latent_model_input, 
                            timestep=t, 
                            encoder_hidden_states=removal_encoder_hidden_states, 
                            feature_mask=removal_feature_mask,
                            masked_feature=removal_masked_feature
                        ).sample # b, 4, 64, 64
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noisy_residual_removal = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64
                                    
                        noisy_residual = noisy_residual_removal
    
                        pred_original_sample = pipe.scheduler.step(noisy_residual, t, removal_latents).pred_original_sample
                        prev_latents = pipe.scheduler.step(noisy_residual, t, removal_latents).prev_sample
                    
                        torch.cuda.empty_cache()
    
                        with torch.no_grad():
                            pred_original_sample_view = pipe.vae.decode(
                                pred_original_sample / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
                            )[0]
                            pred_original_sample_view = pipe.image_processor.postprocess(pred_original_sample_view, output_type="pil", do_denormalize=[True] * pred_original_sample.shape[0])
                            pred_original_sample_view = pred_original_sample_view[0]
                            pred_original_sample_view = np.array(pred_original_sample_view)
                            removal_latents = prev_latents.detach()
    
            # Protocol 1 (for protocol 2, please perform postprocessing by oneself)
            output_image = pipe.vae.decode(
                removal_latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
            )[0]
    
            out_path = os.path.join(out_subdir, "removal.png")
            output_image = pipe.image_processor.postprocess(output_image, output_type="pil", do_denormalize=[True] * output_image.shape[0])
            output_image = output_image[0]
            output_image.save(out_path)
    
            # masked image
            overlay = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(overlay)
    
            mask_polygon = anns['polygon'].copy()
            polygon = [(mask_polygon[i], mask_polygon[i + 1]) for i in range(0, len(mask_polygon), 2)]
            draw.polygon(polygon, fill=(255, 0, 0, 128))
            combined = Image.alpha_composite(image.convert("RGBA"), overlay)
            out_path = os.path.join(out_subdir, "masked.png")
            combined.save(out_path)
        
            output_stack.append(combined)
            output_stack.append(output_image)
        
            new_image = Image.new("RGB", (512*3, 512))
            x_offset = 0
            for img in output_stack:
                new_image.paste(img, (x_offset, 0))
                x_offset += img.width
            out_path = os.path.join(out_subdir, "stitched.png")
            new_image.save(out_path)


if __name__ == "__main__":
    main()
