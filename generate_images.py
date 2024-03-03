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
parser.add_argument('--prompt', type=str, default="A majestic lion jumping from a big stone at night, with star-filled skies. Hyperdetailed, with Complex tropic, African background.")
parser.add_argument('--nprompt', type=str, default="extra limbs")
parser.add_argument('-W','--weight_quant', type=str, default=None)
parser.add_argument('-A','--fwd_quant', type=str, default="M23E8")
parser.add_argument('-f','--flex_bias', action='store_true')
parser.add_argument('-N', '--samples', type=int, default=1)
parser.add_argument('-r', '--repeat_module', type=int, default=1)
parser.add_argument('-R', '--repeat_model', type=int, default=1)
parser.add_argument('--layer_stats', action='store_true')
parser.add_argument('-I', '--individual_care', action='store_true')
parser.add_argument('-i','--inspection', action='store_true')
parser.add_argument('-g','--gamma_threshold', type=float, default=0.001)
parser.add_argument('-q','--quantized_run', action='store_true')
parser.add_argument('-Q','--quantization_noise', type=str, default="expexp")

parser.add_argument('--name', type=str, default="")
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--wandb', action='store_true')


##main

if __name__ == "__main__":

    args = parser.parse_args()

    if args.weight_quant is None:
        args.weight_quant = args.fwd_quant

    torch.cuda.set_device(args.device)

    kwargs = {"quantization_noise": args.quantization_noise, "gamma_threshold": args.gamma_threshold, "quantized_run": args.quantized_run}

    image = run_qpipe(weight_quant = args.weight_quant, weight_flex_bias = args.flex_bias, 
                    fwd_quant = args.fwd_quant, flex_bias = args.flex_bias, 
                    samples=args.samples, n_steps = args.n_steps, name = args.name,
                    repeat_module = args.repeat_module, repeat_model = args.repeat_model, use_wandb=args.wandb,
                    layer_stats = args.layer_stats, individual_care = args.individual_care, inspection = args.inspection,
                    **kwargs)
