import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional
from omegaconf import OmegaConf

import torch
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer,DPTForDepthEstimation,DPTImageProcessor

from tuneavideo.models_wota.unet import UNet3DConditionModel
from tuneavideo.data.dataset import TuneAVideoDataset
from tuneavideo.pipelines.pipeline_pix2video import Pix2VideoPipeline
from tuneavideo.util import save_videos_grid, ddim_inversion
from einops import rearrange


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    train_batch_size: int = 1,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    depth_estimator = DPTForDepthEstimation.from_pretrained(pretrained_model_path, subfolder="depth_estimator")
    feature_extractor = DPTImageProcessor.from_pretrained(pretrained_model_path)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the training dataset
    train_dataset = TuneAVideoDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Get the validation pipeline
    validation_pipeline_depth = Pix2VideoPipeline(
        vae=vae, text_encoder=text_encoder, unet=unet, tokenizer=tokenizer, 
        scheduler=noise_scheduler, depth_estimator=depth_estimator,
        feature_extractor=feature_extractor
    )
    validation_pipeline_depth.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)


    # Prepare everything with our `accelerator`.

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    depth_estimator.to(accelerator.device, dtype=weight_dtype)

    unet = accelerator.prepare(unet)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    if accelerator.is_main_process:
        for step, batch in enumerate(train_dataloader):
            logger.info("inference pixel values")
            pixel_values = batch["pixel_values"].to(accelerator.device,weight_dtype)
            video_length = pixel_values.shape[1]
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215
            pixel_values = (pixel_values+1)*127.5

            samples = []
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)
            ddim_inv_latent = None
            if validation_data.use_inv_latent:
                inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent.pt")
                ddim_inv_latent = ddim_inversion(
                    validation_pipeline_depth, ddim_inv_scheduler, video_latent=latents,
                    num_inv_steps=validation_data.num_inv_steps, prompt="", pixel_values=pixel_values)[-1].to(weight_dtype)
                torch.save(ddim_inv_latent, inv_latents_path)
                # ddim_inv_latent = torch.load(inv_latents_path)
            for idx, prompt in enumerate(validation_data.prompts):
                with torch.autocast("cuda"):
                    sample = validation_pipeline_depth(prompt,pixel_values, generator=generator, latents=ddim_inv_latent,use_l2=True,
                                             **validation_data).videos
                save_videos_grid(sample, f"{output_dir}/samples/sample/depth-{prompt}.gif")
                samples.append(sample)
            samples = torch.concat(samples)
            save_path = f"{output_dir}/samples/depth-sample.gif"
            save_videos_grid(samples, save_path)
            logger.info(f"Saved samples to {save_path}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))