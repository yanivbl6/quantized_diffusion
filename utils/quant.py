import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from QPyTorch.qtorch.quant import *
from math import log2, ceil, floor

from QPyTorch.qtorch import Number, FixedPoint, BlockFloatingPoint, FloatingPoint

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
## torch functional:
import torch.nn.functional as F
##tuple/ optional
from typing import Optional, Tuple

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND

class QAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.enabled:
            key = attn.quantizer(key)
            value = attn.quantizer(value)
            query = attn.quantizer(query)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Quantizer(nn.Module):
    def __init__(
        self,
        forward_number=None,
        backward_number=None,
        forward_rounding="stochastic",
        backward_rounding="stochastic",
        flex_bias=True,
        qdrop=0.0,
    ):
        super(Quantizer, self).__init__()
        self.forward_number = forward_number
        self.backward_number = backward_number
        self.forward_rounding = forward_rounding
        self.backward_rounding = backward_rounding
        self.flex_bias = flex_bias and (forward_number.man <= 8 or forward_number.exp == 0)

        ##import pdb; pdb.set_trace()
        self.quantizer = quantizer(self.forward_number, self.backward_number, self.forward_rounding, self.backward_rounding)

        self.qdrop = qdrop
        self.use_qdrop = qdrop > 0.0
        self.on = True


    def forward(self, x, dim=None):

        mtype = x.dtype

        if mtype == torch.float16:
            x = x.float()

        if not self.on:
            return x

        if self.use_qdrop and self.training:
            x_ = x.clone()
            mask = (torch.rand_like(x) > self.qdrop).float()

        if self.flex_bias:
            e = self.forward_number.exp
            
            assert dim is None, "Flex bias not supported for dim != None"

            if e == 0:

                if len(x.shape) == 4:
                    c = x.abs().max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                elif len(x.shape) == 3:
                    c = x.abs().max(dim=0, keepdim=True)[0]
                elif len(x.shape) == 2:
                    c = x.abs().max(dim=0, keepdim=True)[0]
                else:
                    c = x.abs().max()
                factor = 2**(torch.floor((-torch.log2(c))))
            else:
                m = self.forward_number.man
                c = x.abs().max()/(2 - 2**(-m))
                bhat = 2**(e-1) - log2(c) 
                factor = 2**(floor(bhat))

            x = x*factor
        
        out = self.quantizer.apply(x)
        
        if self.flex_bias:
            out = out/factor
        
        if self.use_qdrop and self.training:
            out = mask*out + (1-mask)*x_

        if mtype == torch.float16:
            out = out.half()

        return out
    
    def enable(self):
        self.on = True

    def disable(self):
        self.on = False

    
def make_weight_quantizer(weights_number: FloatingPoint = FloatingPoint(8, 23), weight_flex_bias: bool = False, stochastic: bool = False):

    if weights_number.exp < 8 or weights_number.man < 23:
        quant = Quantizer(
            weights_number,
            FloatingPoint(8, 23),
            "nearest" if not stochastic else "stochastic", 
            "nearest",
            weight_flex_bias,
            0.0,
        )
    else:
        quant = torch.nn.Identity()


    return quant


def make_block_quantizer(activate: FloatingPoint = FloatingPoint(8, 23),
                         error: FloatingPoint = FloatingPoint(8, 23),
                         activate_rounding: str = "stochastic",
                         error_rounding: str = "stochastic",
                         flex_bias: bool = True,
                         qdrop: float = 0.0,
                         **kwargs):
    
    if activate_rounding == "nearest":
        warnings.warn("Using nearest rounding for activations")

    if activate.exp < 8 or activate.man < 23:
        ##print("Making quantizer with: ", activate, error, activate_rounding, error_rounding, flex_bias, qdrop)
        return Quantizer(
            activate,
            error,
            activate_rounding,
            error_rounding,
            flex_bias,
            qdrop,
            )
    else:
        ##print("Making dummy quantizer")
        return torch.nn.Identity()