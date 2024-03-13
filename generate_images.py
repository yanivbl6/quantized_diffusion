from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
from qunet import QUNet2DConditionModel
from diffusers.models import UNet2DConditionModel
from QPyTorch.qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qpipe import *
import collapse_test

import logging
import wandb

import argparse

parser = argparse.ArgumentParser(description='Run parameters with their default values.')

parser.add_argument('-n','--n_steps', type=int, default=40)
parser.add_argument('--high_noise_frac', type=float, default=0.8)
parser.add_argument('--prompt', type=str, default="lion")
parser.add_argument('-W','--weight_quant', type=str, default=None)
parser.add_argument('-A','--fwd_quant', type=str, default="M23E8")
parser.add_argument('-f','--flex_bias', action='store_true')
parser.add_argument('-N', '--samples', type=int, default=1)
parser.add_argument('-r', '--repeat_module', type=int, default=1)
parser.add_argument('-R', '--repeat_model', type=int, default=1)
parser.add_argument('--layer_stats', action='store_true')
parser.add_argument('-I', '--individual_care', action='store_true')
parser.add_argument('-i','--inspection', action='store_true')
parser.add_argument('-g','--gamma_threshold', type=float, default=1)
parser.add_argument('-q','--quantized_run', action='store_true')
parser.add_argument('-Q','--quantization_noise', type=str, default="cosh")
parser.add_argument('-E', '--use_quantized_euler', action='store_true')
parser.add_argument('--mse', action='store_true')
parser.add_argument('--overwrite', action='store_true')

parser.add_argument('--name', type=str, default="")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--resolution', type=str, default="1024:1024")
parser.add_argument('--include', type=str, default="")
parser.add_argument('-S','--scheduler_noise_mode', type=str, default="dynamic")

parser.add_argument('--wandb', action='store_true')

parser.add_argument('--mode_collapse_experiment', type=float, default=None)

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

    if args.use_quantized_euler:
        args.repeat_module = -1

    if args.mse:
        assert args.repeat_model > 1

    torch.cuda.set_device(args.device)

    kwargs = {"quantization_noise": args.quantization_noise, "gamma_threshold": args.gamma_threshold, "quantized_run": args.quantized_run}

    
    height, width = parse_resolution(args.resolution)


    if args.mode_collapse_experiment is None:

        image = run_qpipe(weight_quant = args.weight_quant, weight_flex_bias = args.flex_bias, 
                        fwd_quant = args.fwd_quant, flex_bias = args.flex_bias, 
                        samples=args.samples, n_steps = args.n_steps, name = args.name,
                        repeat_module = args.repeat_module, repeat_model = args.repeat_model, use_wandb=args.wandb,
                        layer_stats = args.layer_stats, individual_care = args.individual_care, inspection = args.inspection,
                        prompt = args.prompt, high_noise_frac = args.high_noise_frac,
                        calc_mse= args.mse, overwrite = args.overwrite,
                        height = height, width = width, include = args.include,
                        scheduler_noise_mode=args.scheduler_noise_mode,
                        **kwargs)
    
    else:
        collapse_test.run_qpipe(weight_quant = args.weight_quant, weight_flex_bias = args.flex_bias, 
                        fwd_quant = args.fwd_quant, flex_bias = args.flex_bias, 
                        samples=args.samples, n_steps = args.n_steps, name = args.name,
                        repeat_module = args.repeat_module, repeat_model = args.repeat_model, use_wandb=args.wandb,
                        layer_stats = args.layer_stats, individual_care = args.individual_care, inspection = args.inspection,
                        prompt = args.prompt, high_noise_frac = args.high_noise_frac,
                        calc_mse= args.mse, overwrite = args.overwrite,
                        height = height, width = width, include = args.include,
                        scheduler_noise_mode=args.scheduler_noise_mode,
                        alpha = args.mode_collapse_experiment,
                        **kwargs)
