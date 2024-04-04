from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
from qunet import QUNet2DConditionModel
from diffusers.models import UNet2DConditionModel
from QPyTorch.qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qpipe import *

import logging
import wandb

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

##main

def parse_resolution(resolution):
    if resolution is None or resolution == "":
        return 1024, 1024
    elif ":" not in resolution:
        return int(resolution), int(resolution)        

    return tuple(map(int, resolution.split(":")))

if __name__ == "__main__":

    args = parser.parse_args()





    if args.weight_quant is None:
        args.weight_quant = args.fwd_quant

    if args.plus_bits_for_stochastic_weights != 0:

        if args.plus_bits_for_stochastic_weights < 0:
            man = 23
            exp = 8
        else:
            fwd_quant = parse_quant(args.fwd_quant)
            man = fwd_quant.man + args.plus_bits_for_stochastic_weights
            exp = fwd_quant.exp

        args.intermediate_weight_quantization = f"M{man}E{exp}"
        args.stochastic_weights_freq = 1
        args.STEM = 4

    if args.mse:
        assert args.repeat_model > 1

    if args.STEM > 0:
        args.stem = args.STEM + 4

    torch.cuda.set_device(args.device)

    kwargs = {"quantization_noise": args.quantization_noise, "gamma_threshold": args.gamma_threshold, "quantized_run": not args.sim}

    if args.noSR:
        kwargs["activate_rounding"] = "nearest"

    height, width = parse_resolution(args.resolution)

    if args.deterministic:
        ## make everything deterministic
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.n_steps > 0:
        all_steps = [args.n_steps]
    else:
        all_steps = [50, 100, 200, 400, 800]

    for n_steps in all_steps:
        image = run_qpipe(weight_quant = args.weight_quant, weight_flex_bias = args.flex_bias, 
                        fwd_quant = args.fwd_quant, flex_bias = args.flex_bias, 
                        samples=args.samples, n_steps = n_steps, name = args.name,
                        repeat_module = args.repeat_module, repeat_model = args.repeat_model, use_wandb=args.wandb,
                        layer_stats = args.layer_stats, individual_care = args.individual_care, inspection = args.inspection,
                        prompt = args.prompt, high_noise_frac = args.high_noise_frac,
                        calc_mse= args.mse, overwrite = args.overwrite,
                        height = height, width = width, include = args.include,
                        scheduler_noise_mode=args.scheduler_noise_mode,
                        img_directory = args.img_directory, clip_score= args.eval, abort_norm = args.abort_norm,
                        shift_options = args.shift_options, stochastic_emb_mode= args.stem, 
                        stochastic_weights_freq = args.stochastic_weights_freq, 
                        intermediate_weight_quantization = args.intermediate_weight_quantization,
                        dtype = torch.float32 if args.fp32 else torch.float16, prolong= args.prolong,
                        **kwargs)
        
        torch.cuda.empty_cache()
        print("Done with n_steps", n_steps)
