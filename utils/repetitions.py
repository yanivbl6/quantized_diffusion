import torch
import wandb
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput

from diffusers.models.attention_processor import Attention

class Qop(torch.nn.Sequential):
    def __init__(self, quant_op, op):
        super(Qop, self).__init__()
        self.op = op
        self.quant_op = quant_op
        

        self.in_features = self.op.in_features if hasattr(self.op, "in_features") else None
        self.out_features = self.op.out_features if hasattr(self.op, "out_features") else None

    def forward(self, x, scale = 1.0):
        assert scale == 1.0, "Quantized forward pass does not support scaling"
        x = self.quant_op(x)
        x = self.op.forward(x)
        return x

    

class RepeatModule(torch.nn.Module):
    def __init__(self, op, num_iter = 1, name = ""):
        super(RepeatModule, self).__init__()
        self.op = op
        self.num_iter = num_iter
        self.name = name
        self.outtype = None

        self.in_features = self.op.in_features if hasattr(self.op, "in_features") else None
        self.out_features = self.op.out_features if hasattr(self.op, "out_features") else None
    def forward(self, x, *args, **kwargs):

        
        out = self.op.forward(x, *args, **kwargs)
        
        for _ in range(1, self.num_iter):
            out_i = self.op.forward(x, *args, **kwargs)
            if isinstance(out, tuple):
                out = tuple([o + oi for o, oi in zip(out, out_i)])
            else:
                out = out + out_i

        if isinstance(out, tuple):
            out = tuple([o / self.num_iter for o in out])
        else:
            out = out / self.num_iter
    
        return out



class RepeatModuleStats(torch.nn.Module):    
    op = None
    n = 1
    name = ""
    final = False

    def __init__(self, op, num_iter = 1, name = "", final = False, idx = 0):
        super(RepeatModuleStats, self).__init__()
        self.exec_op = op
        self.num_iter = num_iter
        self.mname = name
        self.final_log = final
        self.idx = idx

    def forward(self, x, *args, **kwargs):

    
        out = self.exec_op.forward(x, *args, **kwargs)

        if isinstance(out, tuple):
            out_sum_squares = out[0] ** 2 
        elif isinstance(out, BaseOutput):
            out_sum_squares = out.sample ** 2
        else:
            out_sum_squares = out ** 2

        for _ in range(1, self.num_iter):
            out_i = self.exec_op.forward(x, *args, **kwargs)
            if isinstance(out, tuple):
                out_sum_squares += out_i[0] ** 2
                out = tuple([o + oi for o, oi in zip(out, out_i)])
            elif isinstance(out, BaseOutput):
                out.sample = out.sample + out_i.sample
                out_sum_squares += out.sample ** 2
            else:
                out += out_i
                out_sum_squares += out_i ** 2

        if isinstance(out, tuple):
            out = tuple([o / self.num_iter for o in out])
            variance = ((out_sum_squares / self.num_iter) - out[0] ** 2)
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out[0] ** 2)
        elif isinstance(out, BaseOutput):
            out.sample = out.sample / self.num_iter
            variance = ((out_sum_squares / self.num_iter) - out.sample ** 2)
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out.sample ** 2)
        else:
            out = out / self.num_iter
            variance = ((out_sum_squares / self.num_iter) - out ** 2) 
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out ** 2)
        
        variance = variance.mean()
        standard_deviation = standard_deviation.mean()
        variance_normalized = variance_normalized.mean()
        if wandb.run is not None:
            wandb.log({self.mname + "_variance": variance_normalized.item(), 
                       self.mname + "_Variance": variance.item(),
                       self.mname + "_Std": variance.sqrt().item(),
                       self.mname + "_standard_deviation": standard_deviation.item(),
                       "layer_index": self.idx}, 
                       commit = self.final_log)
                    
        return out

    ## make sure that all other methods are passed to self.op:


    def __getattr__(self, attr):
        if 'exec_op' in self.__dict__:
            return getattr(self.exec_op, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
        
    def __setattr__(self, attr, value):
        if 'exec_op' in self.__dict__:
            setattr(self.exec_op, attr, value)
        else:
            self.__dict__[attr] = value


class HeavyRepeatModule(torch.nn.Module):    
    op = None
    n = 1
    name = ""
    final = False

    def __init__(self, op, num_iter = 1, name = "", final = False, idx = 0, 
                ):## qops_list = [], attn_list = []):
        super(HeavyRepeatModule, self).__init__()
        self.exec_op = op
        self.num_iter = num_iter
        self.mname = name
        self.final_log = final
        self.idx = idx
        ##self.qops_list = qops_list
        ##self.attn_list = attn_list

    def forward(self, x, *args, **kwargs):
    
        out = self.exec_op.forward(x, *args, **kwargs)

        if isinstance(out, tuple):
            out_sum_squares = out[0] ** 2 
        elif isinstance(out, BaseOutput):
            out_sum_squares = out.sample ** 2
        else:
            out_sum_squares = out ** 2

        for _ in range(1, self.num_iter):
            out_i = self.exec_op.forward(x, *args, **kwargs)
            if isinstance(out, tuple):
                out_sum_squares += out_i[0] ** 2
                out = tuple([o + oi for o, oi in zip(out, out_i)])
            elif isinstance(out, BaseOutput):
                out.sample = out.sample + out_i.sample
                out_sum_squares += out.sample ** 2
            else:
                out += out_i
                out_sum_squares += out_i ** 2

        if isinstance(out, tuple):
            out = tuple([o / self.num_iter for o in out])
            variance = ((out_sum_squares / self.num_iter) - out[0] ** 2)
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out[0] ** 2)
        elif isinstance(out, BaseOutput):
            out.sample = out.sample / self.num_iter
            variance = ((out_sum_squares / self.num_iter) - out.sample ** 2)
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out.sample ** 2)
        else:
            out = out / self.num_iter
            variance = ((out_sum_squares / self.num_iter) - out ** 2) 
            standard_deviation = variance.sqrt()
            variance_normalized = variance / (out ** 2)
        
        variance = variance.mean()
        standard_deviation = standard_deviation.mean()
        variance_normalized = variance_normalized.mean()

        ## disable all quantization and attention modules
        unet = self.exec_op

        for name, op in unet.named_modules():
            if isinstance(op, Qop):
                op.quant_op.disable()
            if isinstance(op, Attention):
                op.enabled = False


        ## forward pass with fp32 activations
        out_fp32 = self.exec_op.forward(x, *args, **kwargs)

        ## enable all quantization and attention modules
        for name, op in unet.named_modules():
            if isinstance(op, Qop):
                op.quant_op.enable()
            if isinstance(op, Attention):
                op.enabled = True


        if isinstance(out, tuple):
            mse = ((out_fp32[0] - out[0]) ** 2).mean()
            bias = (out_fp32[0] - out[0])
            bias_out_corr = (bias*out_fp32[0]).mean() / ((out_fp32[0]**2).mean().sqrt() * (bias**2).mean().sqrt())
        elif isinstance(out, BaseOutput):
            mse = ((out_fp32.sample - out.sample) ** 2).mean()
            bias = (out_fp32.sample - out.sample)
            bias_out_corr = (bias*out_fp32.sample).mean() / ((out_fp32.sample**2).mean().sqrt() * (bias**2).mean().sqrt())
        else:
            mse = ((out_fp32 - out) ** 2).mean()
            bias = (out_fp32 - out)
            bias_out_corr = (bias*out_fp32).mean() / ((out_fp32**2).mean().sqrt() * (bias**2).mean().sqrt())

        bias = bias.mean()


        

        if wandb.run is not None:
            wandb.log({self.mname + "_variance": variance_normalized.item(), 
                       self.mname + "_Variance": variance.item(),
                       self.mname + "_Std": variance.sqrt().item(),
                       self.mname + "_standard_deviation": standard_deviation.item(),
                       self.mname + "_corr": bias_out_corr.item(),
                       "MSE": mse.item(),
                       "bias": bias.item(),
                       "layer_index": self.idx}, 
                       commit = self.final_log)
                    
        return out_fp32

    ## make sure that all other methods are passed to self.op:


    def __getattr__(self, attr):
        if 'exec_op' in self.__dict__:
            return getattr(self.exec_op, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
        
    def __setattr__(self, attr, value):
        if 'exec_op' in self.__dict__:
            setattr(self.exec_op, attr, value)
        else:
            self.__dict__[attr] = value

