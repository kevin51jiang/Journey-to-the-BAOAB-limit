
# Gradio File

import gradio as gr

from functools import partial
import os
from pathlib import Path
import random
from typing import Callable, Dict, Optional, Union, List

from diffusers import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from huggingface_hub import notebook_login
from IPython.display import clear_output
import mediapy as media
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import yaml

import torch
assert torch.cuda.is_available()
dtype = torch.float16

from pathlib import Path
from shutil import rmtree

num_samples = 1
device_name = 'cuda'


model_id = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16",
    torch_dtype=dtype)
pipeline = pipeline.to(device_name)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def make_video_stack(images):
  # Make and show a 2x4 grid of videos of the sampling process.
  videos_stack = np.stack(images)  # T, B, 512, 512, 3.
  T, B, H, W, C = videos_stack.shape
  videos_stack = np.transpose(videos_stack, (0, 2, 1, 3, 4))
  videos_stack = np.reshape(videos_stack, (T, H, B * W, C))
  if videos_stack.shape[-1] == 4:
    # Matte with white background.
    videos_stack = (
        videos_stack[..., :3] * videos_stack[..., 3:] +
        (1 - videos_stack[..., 3:]))
  return videos_stack.astype(np.float32)


def sample(
    pipeline: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 200,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    extra_step_kwargs_eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
    run_safety_checker=True,

    method='baoab-limit',  # scheduler, noise-denoise, or baoab-limit
    noise_denoise_eta=2.0,
    fix_t=None,
    t_min=0,
    t_max=None,
    decode_every: int = 5,
):
    with torch.inference_mode():
        # 0. Default height and width to unet
        height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = pipeline._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt
        )
        dtype = text_embeddings.dtype

        # 4. Prepare timesteps
        T = pipeline.scheduler.config.num_train_timesteps
        t_max = min(t_max, T - 1) if t_max else T - 1
        timesteps = np.linspace(
            t_min, t_max, num_inference_steps, dtype=int)[::-1].copy()
        if fix_t:
          timesteps = np.full_like(timesteps, fill_value=fix_t)
        sigmas = np.sqrt(1 - pipeline.scheduler.alphas_cumprod)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(device=device)

        alphas = np.sqrt(pipeline.scheduler.alphas_cumprod)
        alphas = np.interp(timesteps, np.arange(0, len(alphas)), alphas)
        alphas = np.concatenate([alphas, [1.0]]).astype(np.float32)
        alphas = torch.from_numpy(alphas).to(device=device)

        # Copy to device.
        timesteps = torch.from_numpy(timesteps).to(device)
        scheduler.timesteps = timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.in_channels
        def get_noise():
            return pipeline.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        if method == 'baoab-limit':
            eps_prev = get_noise()
            zt = eps_prev * sigmas[0]
        else:
            zt = get_noise()
            eps_prev = zt
        xt = torch.zeros_like(zt)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(
            generator, extra_step_kwargs_eta)

        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            # Get timesteps and coefficients for this sampling iteration.
            alpha_t = alphas[i]
            sigma_t = sigmas[i]

            eps = get_noise()
            if method == 'noise-denoise':
                # The noise-denoise sampler operates in the clean data space rather
                # than diffused data. We replace the latent.
                zt = alpha_t.type(dtype) * xt + sigma_t.type(dtype) * eps

            # expand the latents if we are doing classifier free guidance
            zt_model_input = torch.cat([zt] * 2) if do_classifier_free_guidance else zt
            if method == 'scheduler':
              zt_model_input = pipeline.scheduler.scale_model_input(
                  zt_model_input, t)

            # predict the noise residual
            model_output = pipeline.unet(
                zt_model_input, t.type(dtype),
                encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (
                    model_output_text - model_output_uncond)

            if pipeline.scheduler.config['prediction_type'] == "epsilon":
              epshat = model_output
              xhat = (zt - sigma_t * epshat) / alpha_t
            else:
              raise NotImplementedError

            if i + 1 == timesteps.shape[0]:
                zs = xhat
                xs = xhat
            elif method == 'noise-denoise':
                xs = xt + noise_denoise_eta * sigma_t / alpha_t * (eps - epshat)
                # xs = torch.clamp(xs, -2., 2.)
                zs = zt  # Will be unused.
            else:
                xs = xhat

                if method == 'scheduler':
                    # compute the next noisy sample z_t -> z_s
                    zs = pipeline.scheduler.step(
                        model_output, t, zt, **extra_step_kwargs).prev_sample
                elif method == 'ddim':
                    alpha_s = alphas[i+1]
                    sigma_s = sigmas[i+1]
                    zs = alpha_s * xhat + sigma_s * epshat
                elif method == 'baoab-limit':
                    # remove predicted noise and add back noise correlated
                    # across iterations.
                    zs = zt + sigma_t * (-2 * epshat + eps + eps_prev)

            zt = zs.type(dtype)
            xt = xs.type(dtype)
            eps_prev = eps

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, zt)

            if (i + 1) % decode_every == 0 or (i + 1) == num_inference_steps:
                # To visualize latents:
                # image = xhat.cpu().float().movedim(1, -1).numpy()
                # image = np.clip((image + 1) / 2., 0., 1.)
                # yield image, False

                # 8. Post-processing to visualize pixels
                assert torch.all(torch.isfinite(xhat))
                image = pipeline.decode_latents(xhat)
                assert np.all(np.isfinite(image))

                # 9. Run safety checker
                if run_safety_checker:
                    image, has_nsfw_concept = pipeline.run_safety_checker(
                        image, device, text_embeddings.dtype)
                else:
                    has_nsfw_concept = False

                # 10. Convert to PIL
                if output_type == "pil":
                    image = pipeline.numpy_to_pil(image)

                if not return_dict:
                    yield (image, has_nsfw_concept)
                else:
                    yield StableDiffusionPipelineOutput(
                        images=image, nsfw_content_detected=has_nsfw_concept)



def randomize_seed():
  return random.randint(0, 32767)

with gr.Blocks() as demo:
  # current_image = gr.State(None)

  # Inputs (column)
  with gr.Row():
    with gr.Column():
      prompt = gr.Textbox(label="Enter prompt", placeholder="a DSLR photo of a large basket of rainbow macarons")

      with gr.Row():
        seed = gr.Number(label="Enter seed", value=123321, precision=0)
        btn_randomize = gr.Button("🎲")

      with gr.Accordion("Advanced", open=False):
        #  decode_every=4,
        gr_decode_every = gr.Slider(2, 10, step=1, value=4, label="Decode every n frames", interactive=True)

        # num_inference_steps=200,
        gr_num_inference_steps = gr.Slider(100, 500, value=200, step=1, label="Number of inference steps" , interactive=True)

        # method='baoab-limit'
        gr_method = gr.Dropdown(choices=[ 'noise-denoise', 'baoab-limit'], value='baoab-limit', label="Method") # Removed 'scheduler' because it's causing some bugs
        
        # noise_denoise_eta=2.0,
        gr_noise_denoise_eta=gr_decode_every = gr.Slider(1, 6, 2.0, label="Noise denoise eta", )

        # guidance_scale=3.0,
        gr_guidance_scale=gr_decode_every = gr.Slider(1, 6, 3.0, label="Guidance Scale", )

        # width=512,
        gr_width=gr.Slider(512, 1024, step=1, label="Width (px)", interactive=False)

        # height=512,
        gr_height=gr_width=gr.Slider(512, 1024, step=1, label="Height (px)", interactive=False )

        # fix_t=300
        gr_fix_t = gr.Slider(100, 1000, 300, step=1, label="Fix Time" )

      with gr.Row():
        btn_gen_video = gr.Button("Generate Video")
        btn_cancel_job = gr.Button("Cancel Job")
    with gr.Column():
        completion = gr.Slider(0, gr_num_inference_steps.value, step=1, value=0, label="Iterations", interactive=False)
        output_image = gr.Image(label="Generate Video", visible=True)
        output_video = gr.Video(label="Generate Video", visible=False)




  def do_inference(data):
    num_samples = 1

    if data[prompt] == "":
      data[prompt] = 'a DSLR photo of a large basket of rainbow macarons'
    
    videos = []
    set_seed(int(data[seed]))



    for images, _ in sample(
        pipeline,
        data[prompt] * num_samples,
        return_dict=False,
        run_safety_checker=False,
        output_type="numpy",
        # Sampling arguments.
        decode_every=data[gr_decode_every],
        num_inference_steps=data[gr_num_inference_steps],
        method=data[gr_method],
        noise_denoise_eta=data[gr_noise_denoise_eta],
        guidance_scale=data[gr_guidance_scale],
        width=data[gr_width],
        height=data[gr_height],
        fix_t=data[gr_fix_t]
    ):
        B = images.shape[0]
        videos.append(images)

        images = pipeline.numpy_to_pil(images)
        # clear_output(wait=True)
        # media.show_images(images, width=256)
        out_path = f"/tmp/out-{len(videos)}.png"
        media.write_image(out_path, images[0])
        yield  {
          output_image: gr.update(value=out_path, visible=True),
          output_video: gr.update(value=None, visible=False),
          completion: gr.update(value=len(videos))
        }
    # [f"/tmp/out-{len(videos)}.png", None]

    # Clear all files for next generation


    for path in Path("/tmp").glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)

    media.write_video("/tmp/out.mp4",  make_video_stack(videos), fps=8)
    yield {
      output_image: gr.update(value=None,visible=False),
      output_video: gr.update(value="/tmp/out.mp4", visible=True),
      completion: gr.update(value=1)
    }
    
  def clear_generate_pane():
    return {
        output_image: gr.update(value=None),
        output_video: gr.update(value=None)
    }
  

  btn_randomize.click(
      fn=randomize_seed,
      inputs=None,
      outputs=[seed]
  )

  generate_video_event = btn_gen_video.click(
        fn=do_inference, 
        inputs={prompt, seed, gr_decode_every, gr_num_inference_steps, gr_method, gr_noise_denoise_eta, gr_guidance_scale, gr_width, gr_height, gr_fix_t},
        outputs=[output_image, output_video, completion]
      )
  
  btn_cancel_job.click(
      fn=None,
      inputs=None,
      outputs={output_image, output_video},
      cancels=[generate_video_event]
  )

demo.queue(concurrency_count=2).launch(debug=True, show_api=False)

