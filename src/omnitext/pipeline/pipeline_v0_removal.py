import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageDraw
from typing import Optional
from tqdm.auto import tqdm

from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from omnitext.model.custom_model_v2 import MyUNet2DConditionModel
from omnitext.technique.attend_and_excite import GaussianSmoothing
from omnitext.util.ptp import AttentionStore



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "argmax"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

@dataclass
class FreeTextStyleTransferPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        image (`PIL.Image.Image` or `np.ndarray`)
            denoised PIL image.
    """

    image: Image.Image


class FreeTextStyleTransferPipeline(
    DiffusionPipeline, FromSingleFileMixin
):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    # _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: MyUNet2DConditionModel,
        scheduler: DDPMScheduler
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_image_grid(
        self,               
        image: Image,
    ) -> Image:
        width, height = image.size
        image_array = np.array(image)

        grid_size = 2
        image_grid_array = np.zeros((height * grid_size, width * grid_size, 3), dtype=np.uint8)
        
        image_grid_array[:512, :512, :] = image_array
        image_grid_array[:512, 512:, :] = image_array
        image_grid_array[512:, :512, :] = image_array
        image_grid_array[512:, 512:, :] = image_array
        image_grid = Image.fromarray(image_grid_array)
        return image_grid

    def prepare_image_mask(
        self,
        mask_polygon: tuple[int],
        image_size: tuple[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> (torch.tensor, tuple[int]):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            mask_polygon: (`tuple[int]`)
                polygon coordinate of the mask: x0, y0, x1, y1, x2, y2, x3, y3 (tl, tr, br, bl)
            size: (`tuple[int]`)
                the size of the mask (h, w)
            device: (`torch.device`):
                torch device
            dtype: (`torch.dtype`):
                torch dtype
        """
        h, w = image_size
        image_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(image_mask)

        x0, y0, x1, y1, x2, y2, x3, y3 = mask_polygon

        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=1)

        x0 = x0 // 4
        y0 = y0 // 4
        x1 = x1 // 4
        y1 = y1 // 4
        x2 = x2 // 4
        y2 = y2 // 4
        x3 = x3 // 4
        y3 = y3 // 4
        xmin = min(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        xmax = max(x0, x1, x2, x3)
        ymax = max(y0, y1, y2, y3)
        mask_bbox = (xmin, ymin, xmax, ymax)

        image_mask = torch.tensor(np.array(image_mask)).to(dtype=torch.float32, device=device)
        image_mask = image_mask.unsqueeze(0).unsqueeze(0)

        return image_mask, mask_bbox

    def prepare_curved_image_mask(
        self,
        points,
        bbox,
        image_size: tuple[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> (torch.tensor, tuple[int]):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            mask_polygon: (`tuple[int]`)
                polygon coordinate of the mask: x0, y0, x1, y1, x2, y2, x3, y3 (tl, tr, br, bl)
            size: (`tuple[int]`)
                the size of the mask (h, w)
            device: (`torch.device`):
                torch device
            dtype: (`torch.dtype`):
                torch dtype
        """
        h, w = image_size
        image_mask = np.zeros((h, w), dtype=np.uint8)
        polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(image_mask, [polygon], 255)

        # plt.imshow(image_mask)
        # plt.show()

        # x0, y0, x1, y1 = bbox
        # x0 = x0 // 4
        # y0 = y0 // 4
        # x1 = x1 // 4
        # y1 = y1 // 4
        # mask_bbox = (x0, y0, x1, y1)

        image_mask = torch.tensor(image_mask / 255).to(dtype=torch.float32, device=device)
        image_mask = image_mask.unsqueeze(0).unsqueeze(0)

        return image_mask # , mask_bbox

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        mask_bbox: tuple[int],
        device: torch.device,
        negative_prompt: str = None,
    ) -> (torch.Tensor, torch.Tensor):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            mask_bbox (`tuple[int]`)
                bbox coordinate of the mask: left, top, right, bottom
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
        """
        batch_size = 1 # only support string as prompt

        composed_prompt = ""
        composed_prompt += " <|endoftext|><|startoftext|>"

        per_char_prompt = ' '.join([f'[{c}]' for c in list(prompt)])
        xmin, ymin, xmax, ymax = mask_bbox

        composed_prompt += f' l{xmin} t{ymin} r{xmax} b{ymax} {per_char_prompt} <|endoftext|>'

        text_inputs = self.tokenizer(
            composed_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # print(text_input_ids)
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), 
            attention_mask=attention_mask
        )
        prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        else:
            uncond_tokens = [negative_prompt]
            
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        # TODO: check this code
        uncond_input.input_ids[0][0] = self.tokenizer.pad_token_id
        # print(uncond_input)


        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        return prompt_embeds, negative_prompt_embeds, text_input_ids

    @torch.no_grad()
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator) -> torch.tensor:
        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="sample")
        image_latents = self.vae.config.scaling_factor * image_latents
        return image_latents

    @torch.no_grad()
    def prepare_latents(
        self,
        num_channel_latents: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> torch.tensor:
        shape = (1, num_channel_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = noise * self.scheduler.init_noise_sigma

        return latents

    @torch.no_grad()
    def prepare_inpaint_input(
        self,
        image: Image,
        image_mask: torch.tensor,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> (torch.tensor, torch.tensor):
        width, height = image.size

        image_tensor = self.image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=torch.float32)
        masked_image_tensor = image_tensor * (1 - image_mask)

        masked_image_tensor = masked_image_tensor.to(dtype=dtype)
        # latent = self._encode_vae_image(image_tensor, generator)
        masked_latent = self._encode_vae_image(masked_image_tensor, generator)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        
        latent_mask = F.interpolate(image_mask, size=(latent_height, latent_width), mode="area")
        latent_mask[latent_mask > 0] = 1.0
        latent_mask = latent_mask.to(dtype=dtype)

        return masked_latent, latent_mask


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(
        self, 
        num_inference_steps: int, 
        strength: float, 
        device: torch.device
    ):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    ## ATTEND AND EXCITE ##
    def _aggregate_attention(
        self,
        attention_store: AttentionStore,                
        res: int,                
        from_where: list[str],                
        is_cross: bool,                
        select: int
    ) -> torch.Tensor:
        """ Aggregates the attention across the different layers and heads at the specified resolution. """
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        # print(from_where, num_pixels)
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                # print(item.shape[2], item.shape[1], num_pixels)
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices_to_alter: list[int],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False
    ) -> list[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        # if normalize_eot:
        #     prompt = self.prompt
        #     if isinstance(self.prompt, list):
        #         prompt = self.prompt[0]
        #     last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
    
        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]
    
        # Extract the maximum values
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
            del input
        del smoothing
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(
        self,
        attention_store: AttentionStore,
        indices_to_alter: list[int],
        attention_res: int = 16,
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False
    ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = self._aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        del attention_maps
        return max_attention_per_index

    @staticmethod
    def _compute_loss(
        max_attention_per_index: list[torch.Tensor], 
        return_losses: bool = False
    ) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # print(step_size, type(step_size), type(latents), type(grad_cond))
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: list[int],
        loss: torch.Tensor,
        threshold: float,
        text_embeddings: torch.Tensor,
        text_input,
        latent_mask: torch.Tensor,
        masked_latent: torch.Tensor,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 16,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_refinement_steps: int = 20,
        normalize_eot: bool = False
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1
            
            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings[1].unsqueeze(0), feature_mask=latent_mask, masked_feature=masked_latent).sample
            self.unet.zero_grad()
    
            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
                )
    
            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
    
            if loss != 0:
                # print(loss.requires_grad, latents.requires_grad, step_size)
                latents = self._update_latent(latents, loss, step_size)
    
            with torch.no_grad():
                noise_pred_uncond = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings[0].unsqueeze(0), feature_mask=latent_mask, masked_feature=masked_latent).sample
                noise_pred_text = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings[1].unsqueeze(0), feature_mask=latent_mask, masked_feature=masked_latent).sample
    
            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            # print(indices_to_alter, low_token)
            # print("LTIT", low_token, indices_to_alter[low_token], text_input[indices_to_alter[low_token]])
            # print("HEHE", text_input[indices_to_alter[low_token]])
            low_word = self.tokenizer.decode([text_input[indices_to_alter[low_token]]])
            print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')
    
            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break
    
            torch.cuda.empty_cache()
    
        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings[1].unsqueeze(0), feature_mask=latent_mask, masked_feature=masked_latent).sample
        self.unet.zero_grad()
    
        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        torch.cuda.empty_cache()
        
        return loss, latents, max_attention_per_index

    #######################

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        # latents: torch.Tensor,
        image: Image,
        mask_polygon: tuple[int],
        strength: float = 1.0,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        negative_prompt: str = None,
        generator: torch.Generator = None,
        ## ATTEND AND EXCITE
        controller: AttentionStore = None,
        max_iter_to_alter: int = 20,
        attention_res: int = 32,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 5,
        normalize_eot: bool = False,
        thresholds: dict = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        scale_range: tuple[float] = (1., 0.5),
        **kwargs,
    ):
        image_width, image_height = image.size

        # 0. define parameters
        self._guidance_scale = guidance_scale
        batch_size = 1
        device = self._execution_device
        dtype = self.dtype

        # 1. prepare image grid
        image_grid = self.prepare_image_grid(image)
        grid_width, grid_height = image_grid.size
        grid_size = (grid_height, grid_width)

        # 2. prepare mask
        image_mask, mask_bbox = self.prepare_image_mask(
            mask_polygon,
            grid_size,
            device=device,
            dtype=dtype,
        )
        
        # 3. Encode input prompt
        prompt_embeds_cond, prompt_embeds_uncond, text_input_ids = self.encode_prompt(
            prompt=prompt,
            mask_bbox=mask_bbox,
            device=device,
            negative_prompt=None,
        )

        # 4. Prepare additional input for 
        masked_latent, latent_mask = self.prepare_inpaint_input(
            image=image_grid,
            image_mask=image_mask,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = self.scheduler.timesteps

        # 6. prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents = self.prepare_latents(
            num_channels_latents,
            grid_height,
            grid_width,
            device,
            dtype,
            generator=generator,
        )

        # 7. check the validity of input size
        latents_channel = latents.size(1)
        masked_latent_channel = masked_latent.size(1)
        latent_mask_channel = latent_mask.size(1)
        assert latents_channel + masked_latent_channel + latent_mask_channel == 9
        assert latents.size(2) == masked_latent.size(2) == latent_mask.size(2)
        assert latents.size(3) == masked_latent.size(3) == latent_mask.size(3)

        # 8. attend and excite variables
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        indices_to_alter = list(range(7, 7 + len(prompt)))

        # 9. denoising loop
        encoder_hidden_states = torch.cat([prompt_embeds_uncond, prompt_embeds_cond])
        feature_mask = torch.cat([latent_mask] * 2)
        masked_feature = torch.cat([masked_latent] * 2)

        # return prompt_embeds_uncond, prompt_embeds_cond, latent_mask, masked_latent
        print(indices_to_alter, )
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps), leave=False):
                
                with torch.enable_grad():
                    # latents = latents.clone().detach().requires_grad_(True)
                    latents.requires_grad = True
                    # print(latents.size(), latent_mask.size(), masked_latent.size())
        
                    noise_pred_text = self.unet(sample=latents, timestep=t, encoder_hidden_states=encoder_hidden_states[1].unsqueeze(0), feature_mask=latent_mask, masked_feature=masked_latent).sample
                    self.unet.zero_grad()

                    # Get max activation value for each subject token
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=controller,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=normalize_eot)
        
        
                    loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                    if i in thresholds.keys() and loss > 1. - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                            latents=latents,
                            indices_to_alter=indices_to_alter,
                            loss=loss,
                            threshold=thresholds[i],
                            text_embeddings=encoder_hidden_states,
                            latent_mask=latent_mask,
                            masked_latent=masked_latent,
                            text_input=text_input_ids[0].cpu().numpy(),
                            attention_store=controller,
                            step_size=scale_factor * np.sqrt(scale_range[i]),
                            t=t,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            normalize_eot=normalize_eot)
            
                    # Perform gradient update
                    if i < max_iter_to_alter:
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                        
                        if loss != 0:
                            latents = self._update_latent(latents=latents, loss=loss,
                                                          step_size=scale_factor * np.sqrt(scale_range[i]))
                        print(f'Iteration {i} | Loss: {loss:0.4f}')
        
                    latents = latents.detach()
                    torch.cuda.empty_cache()
        
                latents.requires_grad = False

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                noise_pred = self.unet(sample=latent_model_input, timestep=t, encoder_hidden_states=encoder_hidden_states,feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noisy_residual = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64
                
                latents = self.scheduler.step(noisy_residual, t, latents).prev_sample
                pred_original_sample = self.scheduler.step(noisy_residual, t, latents).pred_original_sample
            
                torch.cuda.empty_cache() 

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False, generator=generator,
        )[0]
        print(image.size())
        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])

        return FreeTextStyleTransferPipelineOutput(image=image)