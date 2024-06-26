from diffusers import DiffusionPipeline, logging, DDIMScheduler, UniPCMultistepScheduler, HeunDiscreteScheduler
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
from utils.safe_mem import GuardMemOp

from PIL import Image



def parse_quant(arg):
    if isinstance(arg, Tuple):
        return FloatingPoint(arg[0], arg[1])
    elif isinstance(arg, FloatingPoint):
        return arg
    elif isinstance(arg, str):
        ## format: "M{mantissa}E{exponent}"

        if arg[0:3].lower() == "int":
            bits = int(arg[3:])

            new_type = FixedPoint(bits, bits-1)
            new_type.exp = 0
            new_type.man = bits - 1
            return  new_type
        else:
            
            match = re.match(r"M(\d+)E(\d+)", arg)
            assert match, "Invalid quantization format"
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
    calc_mse: bool,
    shift_options: int,
    stochastic_emb_mode: int = 0,
    stochastic_weights_freq: int = 0,
    intermediate_weight_quantization: str = "M23E8",
    dtype: torch.dtype = torch.float32,
    prolong: int = 1,
    doubleT: int = 1,
    adjustBN: float = 0.0,
    qstep: int = -1,
    caption_start: int = -1,
    samples: int = 1,
    resolution: Tuple[int, int] = (1024, 1024),
    guidance_scale: float = None,
    eta: float = None,
    scheduler: str = None,
    **kwargs,
) -> str:
    
    name = name + prompt + "x" + str(steps) + "_"

    if guidance_scale is not None:
        name += "gscale_%d_" % int(guidance_scale)

    if eta is not None:
        if eta == 1.0:
            name += "DDPM_"
        elif eta == 0.0:
            name += "DDIM_"
        else:
            name += "eta_%d_" % int(eta*100)

    if scheduler is not None:
        name += scheduler + "_"

    if fwd_quant == weight_quant:
        name = name +  fwd_quant
    else:
        name = name + "A_" + fwd_quant + "_W_" + weight_quant

    if qstep > 0:
        name = name + "_at_step_" + str(qstep)

    if stochastic_weights_freq > 0:
        name = name + "_Wsr"

        if intermediate_weight_quantization != "M23E8":
            name = name + "_" + intermediate_weight_quantization
        
    if stochastic_emb_mode % 4 > 0:
        if stochastic_emb_mode >= 4:
            name = name + "_STEM" + str(stochastic_emb_mode % 4)
        elif stochastic_emb_mode > 0:
            name = name + "_stem" + str(stochastic_emb_mode % 4)

    if not quantized_run and repeat_module < 0:
        name = name + "_sim"

    if not weight_flex_bias:
        name += "_staticBias"

    if include == "" or include == "none" or include == "n":
        name = name + "_noemb"
    elif include != "embedding":
        name = name + "_" + include

    if scheduler_noise_mode != "dynamic":
        name = name + "_SQ_" + scheduler_noise_mode

    if repeat_module > 1:
        name += "_x" + str(repeat_module)
    
    if quantization_noise is not None and quantization_noise.lower() != "none":
        name += "_adjusted"

        if shift_options > 0:
            name += "_shift" + str(shift_options)

        if quantization_noise != "linexp":
            name += "_QN_" + str(quantization_noise)

    if adjustBN != 0.0:
        name += "_BN=%.1E" % adjustBN 

    if prolong > 1:
        name += "_X" + str(prolong)

    if doubleT > 1:
        name += "_eX" + str(doubleT)
    elif doubleT < 0:
        name += "_EX" + str(-doubleT)

    if calc_mse:
        name += "_stats"

    if individual_care:
        name += "_perlayer"

    if repeat_model > 1:
        name += "_N" + str(repeat_model)

    if 'activate_rounding' in kwargs:
        if kwargs['activate_rounding'] == "nearest":
            name += "_nearest" 
        else:
            name += "_rounding_" + kwargs['activate_rounding']

    if dtype == torch.float32:
        name += "_fp32"
    elif dtype == torch.bfloat16:
        name += "_bfloat16"

    if caption_start >= 0 and prompt == "coco":
        name += "_from_" + str(caption_start) + "_to_" + str(caption_start + samples)

    if resolution != (1024, 1024):
        name += "_%dx%d" % resolution


    ## replace "M23E8_StaticBias_fp32" with "_fp32"
    name = name.replace("M23E8_staticBias_fp32", "fp32")
    ## replace "M23E8_StaticBias" with "_fp16"
    name = name.replace("M23E8_staticBias_bfloat16", "bfloat16")
    name = name.replace("M23E8_staticBias", "fp16")
    ## replace "M7E8_StaticBias_fp32" with "_bf16"
    name = name.replace("M7E8_staticBias_fp32", "bf16")
    ## replace "M10E5_StaticBias_fp32" with "_qff16"
    name = name.replace("M10E5_staticBias_fp32", "qff16")

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
              img_directory = "images",
              abort_norm = False,
              shift_options = 0,
              stochastic_emb_mode = 0,
              stochastic_weights_freq = 0,
              intermediate_weight_quantization = "M23E8",
              prolong = 1,
              doubleT = 1,
              adjustBN = 0.0,
              qstep = -1,
              caption_start = -1,
              guidance_scale = None,
              eta = None,
              scheduler = None,
              **kwargs):
    


    quantize_embedding, quantize_first, quantize_last = parse_include(include)


    if prompt == "sd":

        from datasets import load_dataset
        # Load the dataset
        dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')
        # Get the first prompt
        captions = [x['Prompt'] for x in dataset['train']]
        img_idx = list(range(len(captions)))
        nprompt = None
        coco_mode = True

    elif prompt == "coco":
        from T2IBenchmark.datasets import get_coco_30k_captions
        captions = list(get_coco_30k_captions().values())
        img_idx = list(get_coco_30k_captions().keys())
        nprompt = None
        coco_mode = True

    else:
        pprompt, nprompt = get_prompt(prompt)
        coco_mode = False
        




        
    quantization_noise_str = quantization_noise

    if isinstance(fwd_quant, str) and isinstance(weight_quant, str):
        name = base_name(name_or_path, fwd_quant, weight_quant, weight_flex_bias,
                         quantized_run, repeat_module, repeat_model, layer_stats, 
                         individual_care, gamma_threshold, quantization_noise_str, name,  
                         prompt, n_steps, include, scheduler_noise_mode, calc_mse, shift_options, stochastic_emb_mode,
                         stochastic_weights_freq, intermediate_weight_quantization, dtype = dtype , prolong = prolong, 
                         doubleT = doubleT, adjustBN= adjustBN, qstep = qstep, caption_start =caption_start , samples= samples, 
                         resolution = (height, width), guidance_scale = guidance_scale, eta = eta, scheduler = scheduler, **kwargs)
                        

    if caption_start < 0:
        caption_start = 0

    print("-" * 80)
    print("Running: ", name)

    log_dir = "logs"

    with GuardMemOp() as g:
        new_run = True
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(img_directory):
            os.makedirs(img_directory)
        else:
            if os.path.exists(os.path.join(img_directory, name)):
                num_files = len(os.listdir(os.path.join(img_directory, name)))
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

    fwd_quant = parse_quant(fwd_quant)
    weight_quant = parse_quant(weight_quant)
    if stochastic_weights_freq > 0:
        intermediate_weight_quantization = parse_quant(intermediate_weight_quantization)
    else:
        intermediate_weight_quantization = weight_quant 
    qargs = {'activate': fwd_quant}
    ## add kwargs to qargs
    qargs.update(kwargs)


    use_adjusted_scheduler = quantization_noise is not None and quantization_noise != "none"

    if use_adjusted_scheduler:
        base.scheduler = QuantizedEulerDiscreteScheduler.from_scheduler(base.scheduler, 
                                                                        quantization_noise = quantization_noise,
                                                                        gamma_threshold = gamma_threshold,
                                                                        quantized_run = quantized_run,
                                                                        quantization_noise_mode = scheduler_noise_mode,
                                                                        shift_options = shift_options,
                                                                        act_m = fwd_quant.man,
                                                                        inter_m = intermediate_weight_quantization.man,
                                                                        repeat_times= 1)
        
        base.scheduler.set_timesteps(n_steps)
    elif scheduler is not None:
        if scheduler == "UniPC":
            base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
        elif scheduler == "Heun":
            base.scheduler = HeunDiscreteScheduler()
        else:
            raise ValueError("Invalid scheduler")
        
    n_steps1 = n_steps
    
    timestep_to_repetition1 = None



    base.unet = QUNet2DConditionModel.from_unet(base.unet, weight_quant,  weight_flex_bias, qargs, 
                                                repeat_module, repeat_model, layer_stats, individual_care, 
                                                timestep_to_repetition1, calc_mse,
                                                quantize_embedding, quantize_first, quantize_last, abort_norm,
                                                stochastic_emb_mode = stochastic_emb_mode % 4,
                                                stochastic_weights_freq = stochastic_weights_freq,
                                                intermediate_weight_quantization = intermediate_weight_quantization,
                                                adjustBN_scalar = adjustBN, qstep = qstep)

    if stochastic_emb_mode > 4:
        stochastic_emb_mode = 0

    if inspection:
        return base.unet

    if doubleT > 1:
        n_steps2 = n_steps * doubleT
        use_adjusted_scheduler = True
    elif doubleT < 0:
        n_steps2 = n_steps * (-doubleT)
        use_adjusted_scheduler = False
    else:
        n_steps2 = n_steps * prolong

    if use_adjusted_scheduler:
        refiner.scheduler = QuantizedEulerDiscreteScheduler.from_scheduler(refiner.scheduler, 
                                                                           quantization_noise = quantization_noise,
                                                                            gamma_threshold = gamma_threshold,
                                                                            quantized_run = quantized_run,
                                                                            quantization_noise_mode = scheduler_noise_mode,
                                                                            shift_options = shift_options,
                                                                            act_m = fwd_quant.man,
                                                                            inter_m = intermediate_weight_quantization.man,
                                                                            repeat_times= doubleT)

        refiner.scheduler.set_timesteps(n_steps2)
    elif scheduler is not None:
        if scheduler == "UniPC":
            refiner.scheduler = UniPCMultistepScheduler.from_config(refiner.scheduler.config)
        elif scheduler == "Heun":
            base.scheduler = HeunDiscreteScheduler()
        else:
            raise ValueError("Invalid scheduler")
        
    timestep_to_repetition2 = None

    refiner.unet = QUNet2DConditionModel.from_unet(refiner.unet, weight_quant, weight_flex_bias, qargs, 
                                                   repeat_module, repeat_model, layer_stats, individual_care,
                                                   timestep_to_repetition2, calc_mse,
                                                   quantize_embedding, quantize_first, quantize_last, abort_norm,
                                                    stochastic_emb_mode = stochastic_emb_mode % 4,
                                                    stochastic_weights_freq = stochastic_weights_freq,
                                                    intermediate_weight_quantization = intermediate_weight_quantization,
                                                    adjustBN_scalar = adjustBN, qstep = qstep)

    kwargs = {}
    kwargs2 = {}

    if guidance_scale is not None:
        kwargs['guidance_scale'] = guidance_scale

    if eta is not None:
        kwargs['eta'] = eta
        kwargs2['eta'] = eta

    idx = 0



    mimages = []


    generator = torch.Generator(device="cuda")

    subfolder = os.path.join(img_directory, name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)


    logging.set_verbosity(logging.ERROR)

    if samples > 1:
        base.set_progress_bar_config(disable = True)
        refiner.set_progress_bar_config(disable = True)

    pbar = tqdm(range(caption_start,caption_start + samples), desc="Generating images", total = samples)

    for idx in pbar:

        if coco_mode:
            fname = os.path.join(subfolder, "%d.jpeg" % img_idx[idx])
        else:
            fname = os.path.join(subfolder,  "img_%05d.png" % idx)

        if os.path.exists(fname) and not overwrite:
            ## load image from file
            if clip_score and samples > 4:
                image = Image.open(fname)
                mimages.append(image)
            continue


        if use_wandb:
            args= {'prompt': pprompt, 'negative_prompt': nprompt, 'num_inference_steps': n_steps, 'denoising_end': high_noise_frac, 
                    'fwd_quant_e': fwd_quant.exp, 'fwd_quant_m': fwd_quant.man, 'weight_quant_e': weight_quant.exp, 'weight_quant_m': weight_quant.man,
                    'weight_flex_bias': weight_flex_bias, 'dtype': dtype, 'repeat_module': repeat_module, 'repeat_model': repeat_model,
                    'idx': idx, 'version': 6, 'layer_stats': layer_stats, 'individual_care': individual_care, 'inspection': inspection,
                    'gamma_threshold': gamma_threshold, 'quantized_run': quantized_run, "adjusted steps": n_steps1,
                    'scheduler_noise_mode': scheduler_noise_mode, "include": include, "quantization_noise": quantization_noise_str, "abort_norm": abort_norm,
                    'calc_mse': calc_mse, 'shift_options': shift_options, 'rounding': kwargs.get('activate_rounding', 'stochastic'),
                    'stochastic_emb_mode': stochastic_emb_mode, 'stochastic_weights_freq': stochastic_weights_freq, 
                    'intermediate_weight_quantization_man': intermediate_weight_quantization.man, 'intermediate_weight_quantization_exp': intermediate_weight_quantization.exp,
                    'prolong': prolong}
            
            args.update(kwargs)

            ##add kwargs to args
            
            wname = name + "_" + str(idx)

            wandb_entry = wandb.init(project="qpipe", entity= 'dl-projects', name=wname, config=args)


            if not calc_mse:
                use_wandb = False
        else:
            wandb_entry = nullcontext()

        with wandb_entry:
            if prompt == "sd":
                pprompt = captions[idx]
                generator.manual_seed(idx)
            elif coco_mode:
                pprompt = captions[idx]
                generator.manual_seed(42)
            else:
                generator.manual_seed(idx)

            base.unet.step_counter = 0

            image = base(
                prompt=pprompt,
                negative_prompt=nprompt,
                num_inference_steps=n_steps1,
                denoising_end=high_noise_frac,
                output_type="latent",
                generator=generator,
                height = height,
                width = width,
                **kwargs
            ).images


            refiner.unet.step_counter = base.unet.step_counter
            image = refiner(
                prompt=pprompt,
                num_inference_steps=n_steps2,
                denoising_start=high_noise_frac,
                image=image,
                generator=generator,
                height = height,
                width = width,
                **kwargs2
            ).images[0]

        

        ## save the image
            
        
        with GuardMemOp() as g:
            if coco_mode:
                image.save(fname, format = "jpeg")
            else:
                image.save(fname)

        if clip_score and samples > 4:
            mimages.append(image)

    # if samples > 1:

    #     title = name.split("_g_")[0]
    #     title = title.replace("_", " ")
    #     if repeat_module < 0:
    #         title = title + " $\gamma= " + "{:.0e}$".format(gamma_threshold)
    #     plot_grid(name, mimages, title = title)

    if clip_score and samples > 4:

        args= {'prompt': pprompt, 'negative_prompt': nprompt, 'num_inference_steps': n_steps, 'denoising_end': high_noise_frac, 
                'fwd_quant_e': fwd_quant.exp, 'fwd_quant_m': fwd_quant.man, 'weight_quant_e': weight_quant.exp, 'weight_quant_m': weight_quant.man,
                'weight_flex_bias': weight_flex_bias, 'dtype': dtype, 'repeat_module': repeat_module, 'repeat_model': repeat_model,
                'idx': idx, 'version': 6, 'layer_stats': layer_stats, 'individual_care': individual_care, 'inspection': inspection,
                'gamma_threshold': gamma_threshold, 'quantized_run': quantized_run, "adjusted steps": n_steps1,
                'scheduler_noise_mode': scheduler_noise_mode, "include": include, "quantization_noise": quantization_noise_str,
                'shift_options': shift_options}
        
        args.update(kwargs)

        ##add kwargs to args
        
        wname = name + "_" + str(idx)

        ##delete all model to free memory
        del base
        del refiner
        
        torch.cuda.empty_cache()

            
        with wandb.init(project="qpipe_scores", entity= 'dl-projects', name=name, config=args):

            splits = 16 if len(mimages) > 64 else 4
            IS_mean, IS_std = inception_score(mimages, splits = splits) 

            clip_score_mean, clip_score_mean_std  = clip_eval_std(mimages, pprompt, splits = splits)
            clip_score_large_mean, clip_score_large_std  = clip_eval_std(mimages, pprompt, splits = splits, type = "large")
            pickapick_mean, pickapick_std = eval_pickapick(subfolder, pprompt, batch_size=4)

            wandb.log({'count': len(mimages),
                        'clip_score_large': clip_score_large_mean,
                        'clip_score_large_std': clip_score_large_std,
                        'IS': IS_mean,
                        'IS_std': IS_std,
                        'clip_score_mean': clip_score_mean,
                        'clip_score_mean_std': clip_score_mean_std,
                        'pickapick': pickapick_mean,
                        'pickapick_std': pickapick_std,
                       })

            print("CLIP score: ", clip_score_mean, "±", clip_score_mean_std)
            print("CLIP score large: ", clip_score_large_mean, "±", clip_score_large_std)
            print("Inception score: ", IS_mean, "±", IS_std)

    return mimages

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


