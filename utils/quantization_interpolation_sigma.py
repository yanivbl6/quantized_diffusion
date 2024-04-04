import torch
import math
import wandb

import os
import numpy as np

def interpolate_quantization_noise(act_m, inter_m, style, sigmas):

    if act_m >= 7 or style == "zero":
        return 0.0

    n_steps = len(sigmas)


    if inter_m <= act_m:
        paths = {4: "dl-projects/qpipe/vlijm2v0",
                3: "dl-projects/qpipe/z6ks680q",
                2: "dl-projects/qpipe/k3o956cu"}
    else:
        paths = {4: "dl-projects/qpipe/vy542ukr"}

    assert act_m in paths, "Invalid quantization format for automatic noise interpolation"

    path = paths[act_m]

    backup_dir = "objects"

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    fpath = os.path.join(backup_dir,  f"act_M{act_m}_inter_M{inter_m}_" +  str(n_steps) + "_V2.obj")

    if not os.path.exists(fpath) or style == "exact":
        run = wandb.Api().run(path)

        history = run.history()
        qnet_Std = history['qnet_Std']

        if style == "exact":
            assert(n_steps < len(qnet_Std), f"Exact interpolation requested but found only {len(qnet_Std)} steps")
            quant_noise = torch.tensor(qnet_Std.values, dtype=torch.float32)
            print(quant_noise)
            return quant_noise
        
        steps = history['_step']
        q_0 = qnet_Std.values[0]
        q_T = qnet_Std.values[-1]

        torch.save({"q_0": q_0, "q_T": q_T}, fpath)

    else:
        data = torch.load(fpath)
        q_0 = data["q_0"]
        q_T = data["q_T"]

    
    if style == "linexp":
        quant_noise = linexp(sigmas, q_0, q_T)
    else:
        raise ValueError("Invalid style for automatic noise interpolation")
    
    return quant_noise


def linexp(sigmas, q_0, q_T):
    a = np.log(q_0/q_T)/np.log(sigmas[0]/sigmas[-2])
    b = np.exp(np.log(q_0) - a*np.log(sigmas[0]))
    q_std_hat = b*sigmas**(a)
    q_std_hat = torch.tensor(q_std_hat[:-1], dtype=torch.float32)
    return q_std_hat