from T2IBenchmark import calculate_coco_fid
from T2IBenchmark.models.kandinsky21 import Kandinsky21Wrapper

import os

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
from utils.safe_mem import GuardMemOp

from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Run parameters with their default values.')

parser.add_argument('-n','--n_steps', type=int, default=40)
parser.add_argument('--high_noise_frac', type=float, default=0.8)
parser.add_argument('--prompt', type=str, default="morgana2")
parser.add_argument('-W','--weight_quant', type=str, default=None)
parser.add_argument('-A','--fwd_quant', type=str, default="M23E8")
parser.add_argument('-f','--flex_bias', action='store_true')
parser.add_argument('-N', '--samples', type=int, default=64)
parser.add_argument('-r', '--repeat_module', type=int, default=1)
parser.add_argument('-R', '--repeat_model', type=int, default=1)
parser.add_argument('--layer_stats', action='store_true')
parser.add_argument('-I', '--individual_care', action='store_true')
parser.add_argument('-i','--inspection', action='store_true')
parser.add_argument('-g','--gamma_threshold', type=float, default=1)
parser.add_argument('--sim', action='store_true')
parser.add_argument('-Q','--quantization_noise', type=str, default="none")

parser.add_argument('--mse', action='store_true')
parser.add_argument('--abort_norm', action='store_true')



parser.add_argument('--overwrite', action='store_true')

parser.add_argument('--eval', action='store_true')


parser.add_argument('--name', type=str, default="")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--resolution', type=str, default="1024:1024")
parser.add_argument('--include', type=str, default="embedding")
parser.add_argument('-S','--scheduler_noise_mode', type=str, default="dynamic")

parser.add_argument('--wandb', action='store_true')
parser.add_argument('--shift_options', type=int, default=0)

parser.add_argument('--noSR', action='store_true')

parser.add_argument('--deterministic', action='store_true')


parser.add_argument('--img_directory', type=str, default="images")

parser.add_argument('--stem', type=int, default=0)
parser.add_argument('--STEM', type=int, default=0)
parser.add_argument('--stochastic_weights_freq', type=int, default=0)

parser.add_argument('--intermediate_weight_quantization', type=str, default="M23E8")

parser.add_argument('-p','--plus_bits_for_stochastic_weights', type=int, default=0)

parser.add_argument('--fp32', action='store_true')


parser.add_argument('-x','--prolong', type=int, default=1)
parser.add_argument('-X','--doubleT', type=int, default=1)


parser.add_argument('--bn', type=float, default=0.0)

parser.add_argument('--qstep', type=int, default=-1)

parser.add_argument('--dir_name', type=str, default="misc")



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
    **kwargs,
) -> str:
    
    name = name + prompt + "x" + str(steps) + "_"

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





def get_qpipe(name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
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
              **kwargs):
    

    quantize_embedding, quantize_first, quantize_last = parse_include(include)

    pprompt, nprompt = get_prompt(prompt)



    quantization_noise_str = quantization_noise

    if isinstance(fwd_quant, str) and isinstance(weight_quant, str):
        name = base_name(name_or_path, fwd_quant, weight_quant, weight_flex_bias,
                         quantized_run, repeat_module, repeat_model, layer_stats, 
                         individual_care, gamma_threshold, quantization_noise_str, name,  
                         prompt, n_steps, include, scheduler_noise_mode, calc_mse, shift_options, stochastic_emb_mode,
                         stochastic_weights_freq, intermediate_weight_quantization, dtype = dtype , prolong = prolong, 
                         doubleT = doubleT, adjustBN= adjustBN, qstep = qstep, **kwargs) 

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

    
    timestep_to_repetition2 = None

    refiner.unet = QUNet2DConditionModel.from_unet(refiner.unet, weight_quant, weight_flex_bias, qargs, 
                                                   repeat_module, repeat_model, layer_stats, individual_care,
                                                   timestep_to_repetition2, calc_mse,
                                                   quantize_embedding, quantize_first, quantize_last, abort_norm,
                                                    stochastic_emb_mode = stochastic_emb_mode % 4,
                                                    stochastic_weights_freq = stochastic_weights_freq,
                                                    intermediate_weight_quantization = intermediate_weight_quantization,
                                                    adjustBN_scalar = adjustBN, qstep = qstep)
    
    return base, refiner, n_steps1, n_steps2, name




import torch
from PIL import Image
from T2IBenchmark import T2IModelWrapper
from kandinsky2 import get_kandinsky2

def parse_resolution(resolution):
    if resolution is None or resolution == "":
        return 1024, 1024
    elif ":" not in resolution:
        return int(resolution), int(resolution)        

    return tuple(map(int, resolution.split(":")))

class QPIPE(T2IModelWrapper):
    
    def load_model(self, device, save_dir="misc_generation", use_saved_images=True, seed=0):
        args = parser.parse_args()

        base_dir = "FID_images"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.save_dir = os.path.join(base_dir, args.dir_name)
        self.use_saved_images = use_saved_images
        self.seed = seed

        if args.weight_quant is None:
            args.weight_quant = args.fwd_quant

        if args.plus_bits_for_stochastic_weights != 0:

            if args.plus_bits_for_stochastic_weights < 0:
                man = 23
                exp = 8
            else:
                weight_quant = parse_quant(args.weight_quant)
                man = weight_quant.man + args.plus_bits_for_stochastic_weights
                exp = weight_quant.exp

            args.intermediate_weight_quantization = f"M{man}E{exp}"
            args.stochastic_weights_freq = 1
            args.STEM = 4

        if args.mse:
            assert args.repeat_model > 1

        if args.STEM > 0:
            args.stem = args.STEM + 4

        if args.doubleT != 1 and args.quantization_noise == "none":
            args.quantization_noise = "zero"


        torch.cuda.set_device(device)

        kwargs = {"quantization_noise": args.quantization_noise, "gamma_threshold": args.gamma_threshold, "quantized_run": not args.sim}

        if args.noSR:
            kwargs["activate_rounding"] = "nearest"

        height, width = parse_resolution(args.resolution)

        if args.deterministic:
            ## make everything deterministic
            torch.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.args = args
        self.kwargs = kwargs
        self.n_steps = args.n_steps
        self.height = height
        self.width = width

        """Initialize model here"""
        kwargs = self.kwargs
        args = self.args
        n_steps = self.n_steps

        self.base, self.refiner, self.n_steps1, self.n_steps2, self.name = get_qpipe(weight_quant = args.weight_quant, weight_flex_bias = args.flex_bias, 
                        fwd_quant = args.fwd_quant, flex_bias = args.flex_bias, 
                        samples=args.samples, n_steps = n_steps, name = args.name,
                        repeat_module = args.repeat_module, repeat_model = args.repeat_model, use_wandb=args.wandb,
                        layer_stats = args.layer_stats, individual_care = args.individual_care, inspection = args.inspection,
                        prompt = args.prompt, high_noise_frac = args.high_noise_frac,
                        calc_mse= args.mse, overwrite = args.overwrite,
                        height = self.height, width = self.width, include = args.include,
                        scheduler_noise_mode=args.scheduler_noise_mode,
                        img_directory = args.img_directory, clip_score= args.eval, abort_norm = args.abort_norm,
                        shift_options = args.shift_options, stochastic_emb_mode= args.stem, 
                        stochastic_weights_freq = args.stochastic_weights_freq, 
                        intermediate_weight_quantization = args.intermediate_weight_quantization,
                        dtype = torch.float32 if args.fp32 else torch.float16, prolong= args.prolong,
                        doubleT = args.doubleT, adjustBN = args.bn, qstep = args.qstep,
                        **kwargs)
        
        self.high_noise_frac = args.high_noise_frac

        self.generator = torch.Generator(device="cuda")

        self.name = "MSCOCO_FID_" + self.name

        self.generator.manual_seed(seed)
        self.base.set_progress_bar_config(disable = True)
        self.refiner.set_progress_bar_config(disable = True)

    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""
        image = self.base(
            prompt=caption,
            num_inference_steps=self.n_steps1,
            generator=self.generator,
            denoising_end=self.high_noise_frac,
            output_type="latent",
            height = self.height,
            width = self.width,
        ).images
        image = self.refiner(
            prompt=caption,
            num_inference_steps=self.n_steps2,
            generator=self.generator,
            denoising_start=self.high_noise_frac,
            image=image,
            height = self.height,
            width = self.width,
        ).images[0]

        return image
    
if __name__ == "__main__":


    fid, fid_data = calculate_coco_fid(
    QPIPE,
    device='cuda:0',
    save_generations_dir="misc_generation",
)