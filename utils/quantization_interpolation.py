import torch
import math
import wandb

import os

def interpolate_quantization_noise(fwd_quant, style, n_steps, include = ""):

    include="" ## remove me

    if fwd_quant == "M23E8" or fwd_quant == "M10E5" or fwd_quant == "M7E8":
        return 0.0

    if include == "":
        if n_steps == 40:
            paths = {"M4": "dl-projects/qpipe/m22a82qa",
                    "M3": "dl-projects/qpipe/9pcbity4",
                    "M2": "dl-projects/qpipe/hiidxu4f"}
        elif n_steps == 30:
            paths = {"M4": "dl-projects/qpipe/8rrb80yt",
                    "M3": "dl-projects/qpipe/bca7qs1m",
                    "M2": "dl-projects/qpipe/7xo6mgmm"}
        else:
            paths = {"M4": "dl-projects/qpipe/251jhla0",
                    "M3": "dl-projects/qpipe/h5w65f23",
                    "M2": "dl-projects/qpipe/gpamc6dx"}
    else:
        paths = {"M4": "dl-projects/qpipe/urtdzovp",
                "M3": "dl-projects/qpipe/e5teob1d",
                "M2": "dl-projects/qpipe/cfsbrumt"}

    fwd_quant_m = fwd_quant.split("E")[0]

    assert fwd_quant_m in paths, "Invalid quantization format for automatic noise interpolation"

    path = paths[fwd_quant_m]

    backup_dir = "objects"

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    if include != "":
        fpath = os.path.join(backup_dir, fwd_quant + "_" +  str(n_steps) + "_" + include + ".obj")
    else:
        fpath = os.path.join(backup_dir, fwd_quant + "_" +  str(n_steps) + ".obj")

    if not os.path.exists(fpath):
        run = wandb.Api().run(path)

        history = run.history()
        qnet_Std = history['qnet_Std']
        steps = history['_step']
        T = len(steps)
        y0 = qnet_Std.values[0]
        y1 = qnet_Std.values[-1]
        ymid = qnet_Std.values[T//2]

        torch.save({"y0": y0, "y1": y1, "T": T, "ymid": ymid}, fpath)
        T = n_steps

    else:
        data = torch.load(fpath)
        y0 = data["y0"]
        ymid = data["ymid"]
        y1 = data["y1"]

        T = n_steps
    
    if style == "expexp":
        quant_noise = expexp_quantization_noise(y0, y1, T, ymid)
    elif style == "exp":
        quant_noise = exponential_quantization_noise(y0, y1, T)
    elif style == "poly":
        quant_noise = polynomial_quantization_noise(y0, y1, T, ymid)
    elif style == "cosh":
        quant_noise = cosh_quantization_noise(y0, y1, T, ymid)
    else:
        raise ValueError("Invalid style for automatic noise interpolation")
    
    return quant_noise

def exponential_quantization_noise(q0, q1, T):
    x = torch.arange(0, T, dtype=torch.float32)
    beta = math.log(q1/q0)/(T-1)
    y = q0 * torch.exp(beta*x)
    return y

def expexp_quantization_noise(q0, q1, T):
    x = torch.arange(0, T, dtype=torch.float32)
    ## beta = \frac{1}{T}\log\left(\log\left(\frac{y_{1}}{y_{0}}\right)+1\right)
    beta = math.log(math.log(q1/q0)+1)/(T-1)
    y = q0 * torch.exp(torch.exp(beta*x) - 1) 
    return y

def cosh_quantization_noise(q0, q1, T, qmid):
    x = torch.arange(0, T, dtype=torch.float32)    
    a = q0
    b = ((q0/q1)**2-1)/(T-1)**2
    y = a / torch.sqrt(b*x**2+1) 
    return y


def expexp_quantization_noise(q0, qT, T, qmid = None):


    x = torch.arange(0, T, dtype=torch.float32)
    ## beta = \frac{1}{T}\log\left(\log\left(\frac{y_{1}}{y_{0}}\right)+1\right)
    
    if qmid is None:
        b = math.log(math.log(qT/q0)+1)/(T-1)
        y = q0 * torch.exp(torch.exp(b*x) - 1) 
    else:
        
        a = q0
        ## c=-\frac{2}{T}\log\left(\frac{\log\left(\frac{y_{1}}{y_{0}}\right)}{\log\left[\frac{y_{2}}{y_{1}}\right]}\right)
        c = -2*math.log(math.log(qmid/q0)/math.log(qT/qmid))/(T-1)
        ## b=\frac{\log\left[\frac{y_{2}}{y_{1}}\right]}{\left(\exp\left[cT\right]-\exp\left[c\frac{T}{2}\right]\right)}
        b = math.log(qT/qmid)/(math.exp(c*(T-1))-math.exp(c*(T//2)))
        y= a * torch.exp(b* (torch.exp(c*x) - 1))

    return y

def polynomial_quantization_noise(q0, q1, T, qmid = None):
    x = torch.arange(0, T, dtype=torch.float32)
    if qmid is None:
        b = 0
        a = (q1 - q0)/(T-1)**2
        c = q0
    else:
        c = q0
        ##\frac{2\left(y_{2}+y_{0}\right)-4y_{1}}{T^{2}}=a
        a = (2*(q1+q0)-4*qmid)/(T-1)**2
        ##\frac{4y_{1}-y_{2}-3y_{0}}{T}=b
        b = (4*qmid-q1-3*q0)/(T-1)

    y = a*x**2 + b*x + c
    return y