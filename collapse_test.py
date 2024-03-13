from diffusers import DiffusionPipeline, logging
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from qunet import QUNet2DConditionModel
from quantized_euler_discrete import QuantizedEulerDiscreteScheduler

from diffusers.models import UNet2DConditionModel


from QPyTorch.qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint

from typing import Tuple

import os
import re
import wandb
import math
from utils.quantization_interpolation import interpolate_quantization_noise
from utils.presentation import plot_grid
from utils.evaluate import *
from utils.prompts import get_prompt

from PIL import Image

def parse_quant(arg):
    if isinstance(arg, Tuple):
        return FloatingPoint(arg[0], arg[1])
    elif isinstance(arg, FloatingPoint):
        return arg
    elif isinstance(arg, str):
        ## format: "M{mantissa}E{exponent}"
        match = re.match(r"M(\d+)E(\d+)", arg)
        m = int(match.group(1))
        e = int(match.group(2))
        return FloatingPoint(e, m)
    else:
        raise ValueError("Invalid quantization format")


from contextlib import nullcontext, redirect_stdout


def base_name(
    name_or_path: str,
    fwd_quant: str,
    weight_quant: str,
    weight_flex_bias: bool,
    quantized_run: bool,
    repeat_module: int,
    repeat_model: int,
    layer_stats: bool,
    individual_care: bool,
    gamma_threshold: float,
    quantization_noise: float,
    name: str,
    prompt: str,
    steps: int,
    include: str,
    scheduler_noise_mode: str,
    **kwargs,
) -> str:
    
    name = name + prompt + "x" + str(steps) + "_"

    if fwd_quant == weight_quant:
        name = name +  fwd_quant
    else:
        name = name + "A_" + fwd_quant + "_W_" + weight_quant

    if not quantized_run and repeat_module < 0:
        name = "sim_" + name

    if weight_flex_bias:
        name += "_flex"

    if include != "" and include != "none" and include != "n":
        name = name + "_" + include

    if scheduler_noise_mode != "dynamic":
        name = name + "_SQ_" + scheduler_noise_mode

    if repeat_module > 1:
        name += "_x" + str(repeat_module)
    elif repeat_module < 0:
        name += "_adjustedV2"



    elif repeat_module < 0:
        name += "_dynamic"

    if individual_care:
        name += "_perlayer"

    if repeat_model > 1:
        name += "_N" + str(repeat_model)


    

    return name


def parse_include(option: str):
    option = option.lower()
    quantize_embedding = False
    quantize_first = False
    quantize_last = False

    if option in ["all", "a"]:
        quantize_embedding = True
        quantize_first = True
        quantize_last = True
    elif option in ["first", "f"]:
        quantize_first = True
    elif option in ["last", "l"]:
        quantize_last = True
    elif option in ["embedding", "emb", "e"]:
        quantize_embedding = True
    elif option in ["none", "n", ""]:
        pass
    elif option in ["not_first", "nf"]:
        quantize_embedding = True
        quantize_last = True
    elif option in ["not_last", "nl"]:
        quantize_embedding = True
        quantize_first = True
    elif option in ["not_embedding", "ne"]:
        quantize_first = True
        quantize_last = True
    else:
        raise ValueError("Invalid include option")

    return quantize_embedding, quantize_first, quantize_last


from utils.repetitions import Qop
from diffusers.models.attention_processor import Attention

def set_quantization(unet, enabled):
    for name, op in unet.named_modules():
        if isinstance(op, Qop):
            if enabled:
                op.quant_op.enable()
            else:
                op.quant_op.disable()
        if isinstance(op, Attention):
            op.enabled = enabled



def run_qpipe(name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
              name_or_path_ref = "stabilityai/stable-diffusion-xl-refiner-1.0",
              n_steps = 40,
              high_noise_frac = 0.8,
              prompt = "lion",
              fwd_quant = FloatingPoint(8, 23),
              weight_quant = FloatingPoint(8, 23),
              weight_flex_bias = False,
              dtype = torch.float32,
              name = "",
              samples = 1,
              repeat_module = 1,
              repeat_model = 1,
              layer_stats = False,
              individual_care = False,
              inspection = False,
              quantization_noise = "cosh",
              gamma_threshold = 1.0,
              quantized_run = True,
              use_wandb = True,
              clip_score = True,
              calc_mse = False,
              overwrite = False,
              height = 1024,
              width = 1024,
              include= "",
              scheduler_noise_mode = "dynamic",
              alpha = 0.0,
              **kwargs):
    

    quantize_embedding, quantize_first, quantize_last = parse_include(include)

    pprompt, nprompt = get_prompt(prompt)

    quantization_noise_str = ""
    if repeat_module< 0 and isinstance(quantization_noise, str):
        quantization_noise_str = quantization_noise
        quantization_noise = interpolate_quantization_noise(fwd_quant, quantization_noise, n_steps, include=include)

    if isinstance(fwd_quant, str) and isinstance(weight_quant, str):
        name = base_name(name_or_path, fwd_quant, weight_quant, weight_flex_bias,
                         quantized_run, repeat_module, repeat_model, layer_stats, 
                         individual_care, gamma_threshold, quantization_noise, name,  
                         prompt, n_steps, include, scheduler_noise_mode,
                         **kwargs) 

    print("-" * 80)
    print("Running: ", name)

    log_dir = "logs"

    new_run = True
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else: 
        num_files = len([name for name in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, name))])
        print("exists with ", num_files, " files")
        new_run = False

    print("-" * 80)


    log_file = os.path.join(log_dir, name + ".log")

    if not quantized_run and repeat_module < 0:
        fwd_quant = "M23E8"


    torch.cuda.empty_cache()


    base = DiffusionPipeline.from_pretrained(
        name_or_path, torch_dtype=dtype, use_safetensors=True,
        variant = "fp16" if dtype == torch.float16 else None, 
    )
    base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        name_or_path_ref,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant = "fp16" if dtype == torch.float16 else None, 
    )

    refiner.to("cuda")

    if repeat_module < 0:
        # name = name + "_g_" + "{:.0e}".format(gamma_threshold)
        base.scheduler = QuantizedEulerDiscreteScheduler.from_scheduler(base.scheduler, 
                                                                        quantization_noise = quantization_noise,
                                                                        gamma_threshold = gamma_threshold,
                                                                        quantized_run = quantized_run,
                                                                        quantization_noise_mode = scheduler_noise_mode)
        
        base.scheduler.set_timesteps(n_steps)


        n_steps1 = n_steps
        # n_steps1 = base.scheduler.get_adjusted_timesteps()
        
        # base.scheduler = QuantizedEulerDiscreteScheduler.from_scheduler(base.scheduler, 
        #                                                                 quantization_noise = quantization_noise,
        #                                                                 gamma_threshold = gamma_threshold,
        #                                                                 quantized_run = quantized_run)

        # base.scheduler.set_timesteps(n_steps1)

        timestep_to_repetition1 = base.scheduler.make_repetition_plan()


    else:
        n_steps1 = n_steps
        timestep_to_repetition1 = None

    fwd_quant = parse_quant(fwd_quant)
    weight_quant = parse_quant(weight_quant)
    qargs = {'activate': fwd_quant}
    ## add kwargs to qargs
    qargs.update(kwargs)

    base.unet = QUNet2DConditionModel.from_unet(base.unet, weight_quant,  weight_flex_bias, qargs, 
                                                repeat_module, repeat_model, layer_stats, individual_care, 
                                                timestep_to_repetition1, calc_mse,
                                                quantize_embedding, quantize_first, quantize_last)

    if inspection:
        return base.unet

    if repeat_module < 0:
        refiner.scheduler = QuantizedEulerDiscreteScheduler.from_scheduler(refiner.scheduler, 
                                                                           quantization_noise = quantization_noise,
                                                                            gamma_threshold = gamma_threshold,
                                                                            quantized_run = quantized_run,
                                                                            quantization_noise_mode = scheduler_noise_mode)
        refiner.scheduler.set_timesteps(n_steps1)
        timestep_to_repetition2 = refiner.scheduler.make_repetition_plan()

    else:
        timestep_to_repetition2 = None

    refiner.unet = QUNet2DConditionModel.from_unet(refiner.unet, weight_quant, weight_flex_bias, qargs, 
                                                   repeat_module, repeat_model, layer_stats, individual_care,
                                                   timestep_to_repetition2, calc_mse,
                                                   quantize_embedding, quantize_first, quantize_last)


    idx = 0

    if not os.path.exists("collapse"):
        os.makedirs("collapse")

    mimages = []


    generator = torch.Generator(device="cuda")

    subfolder = os.path.join("collapse", name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)


    logging.set_verbosity(logging.ERROR)

    base.set_progress_bar_config(disable = True)
    refiner.set_progress_bar_config(disable = True)

    diff_f_q = 0
    diff_f_f = 0
    diff_q_q = 0
    pbar = tqdm(range(samples), desc="Generating images", total = samples)

    if use_wandb:

        args= {'prompt': pprompt, 'negative_prompt': nprompt, 'num_inference_steps': n_steps, 'denoising_end': high_noise_frac, 
                'fwd_quant_e': fwd_quant.exp, 'fwd_quant_m': fwd_quant.man, 'weight_quant_e': weight_quant.exp, 'weight_quant_m': weight_quant.man,
                'weight_flex_bias': weight_flex_bias, 'dtype': dtype, 'repeat_module': repeat_module, 'repeat_model': repeat_model,
                'idx': idx, 'version': 5, 'layer_stats': layer_stats, 'individual_care': individual_care, 'inspection': inspection,
                'gamma_threshold': gamma_threshold, 'quantized_run': quantized_run, "adjusted steps": n_steps1,
                'scheduler_noise_mode': scheduler_noise_mode, "include": include, "quantization_noise": quantization_noise_str,
                "alpha": alpha}
        
        args.update(kwargs)
        wandb_entry = wandb.init(project="qpipe_collapse", entity= 'dl-projects', name=name, config=args)
        use_wandb = False
    else:
        wandb_entry = nullcontext()



    with wandb_entry:
        for idx in pbar:
            generator.manual_seed(idx)
            init_latent = torch.randn(1, 4, height//8, width//8, device="cuda")

            generator.manual_seed(idx)
            latent = base(
                prompt=pprompt,
                negative_prompt=nprompt,
                num_inference_steps=n_steps1,
                denoising_end=high_noise_frac,
                latents=init_latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            latent = refiner(
                prompt=pprompt,
                num_inference_steps=n_steps1,
                denoising_start=high_noise_frac,
                image=latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            latent_quantized = latent.clone().detach().cpu()

            set_quantization(base.unet, False)
            generator.manual_seed(idx)
            latent = base(
                prompt=pprompt,
                negative_prompt=nprompt,
                num_inference_steps=n_steps1,
                denoising_end=high_noise_frac,
                latents=init_latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            latent = refiner(
                prompt=pprompt,
                num_inference_steps=n_steps1,
                denoising_start=high_noise_frac,
                image=latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            latent_fp32 = latent.clone().detach()

            latent_quantized = latent_quantized.cuda()

            diff_direction = latent_quantized - latent_fp32

        
            diff_direction =  (init_latent.norm() / diff_direction.norm()) * diff_direction


            new_init_latent = init_latent + (1-math.fabs(alpha)) + alpha * diff_direction

            latent_quantized = latent_quantized.cpu()
            latent_fp32 = latent_fp32.cpu()

            set_quantization(base.unet, True)
            generator.manual_seed(idx)
            latent = base(
                prompt=pprompt,
                negative_prompt=nprompt,
                num_inference_steps=n_steps1,
                denoising_end=high_noise_frac,
                latents=new_init_latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            generator.manual_seed(idx)
            latent = refiner(
                prompt=pprompt,
                num_inference_steps=n_steps1,
                denoising_start=high_noise_frac,
                image=latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            diff_q_q += ((latent - latent_quantized.cuda())**2).mean().item()

            latent_quantized = latent.clone().detach().cpu()


            set_quantization(base.unet, False)
            generator.manual_seed(idx)
            latent = base(
                prompt=pprompt,
                negative_prompt=nprompt,
                num_inference_steps=n_steps1,
                denoising_end=high_noise_frac,
                latents=new_init_latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            generator.manual_seed(idx)
            latent = refiner(
                prompt=pprompt,
                num_inference_steps=n_steps1,
                denoising_start=high_noise_frac,
                image=latent,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
            ).images

            diff_f_f += ((latent - latent_fp32.cuda())**2).mean().item()

            latent_fp32 = latent.clone().detach()


            diff_f_q += ((latent_quantized.cuda() - latent_fp32)**2).mean().item()

            pbar.set_postfix({"diffs: f-q": diff_f_q/(idx+1), "diffs f-f": diff_f_f/(idx+1), "diffs q-q": diff_q_q/(idx+1)})



            if wandb.run is not None:
                wandb.log({'diffs f-q': diff_f_q, 'diffs f-f': diff_f_f, 'diffs q-q': diff_q_q})

        if wandb.run is not None:
            wandb.log({'total diffs f-q': diff_f_q/samples, 
                       'total diffs f-f': diff_f_f/samples, 
                       'total diffs q-q': diff_q_q/samples})




    return diff/samples

def get_qnet(name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
              name_or_path_ref = "stabilityai/stable-diffusion-xl-refiner-1.0",
              fwd_quant = FloatingPoint(8, 23),
              weight_quant = FloatingPoint(8, 23),
              weight_flex_bias = False,
              dtype = torch.float32,
              repeat_module = 1,
              repeat_model = 1,
              **kwargs):
    
    fwd_quant = parse_quant(fwd_quant)
    weight_quant = parse_quant(weight_quant)

    torch.cuda.empty_cache()
    qargs = {'activate': fwd_quant}
    ## add kwargs to qargs
    qargs.update(kwargs)

    base = DiffusionPipeline.from_pretrained(
        name_or_path, torch_dtype=dtype, use_safetensors=True,
        variant = "fp16" if dtype == torch.float16 else None, 
    )
    base.to("cuda")

    unet = QUNet2DConditionModel.from_unet(base.unet, weight_quant,  weight_flex_bias, qargs, repeat_module, repeat_model)
    del base

    return unet


