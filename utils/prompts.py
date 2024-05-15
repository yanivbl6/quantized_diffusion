from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import os

promptsub = {
    "*epic*": "Epic Realistic, high Quality, masterpiece, neutral colors, screen space refractions, (intricate details, hyperdetailed:1.2), artstation, complex background",
}

promptdict = {
    "morgana2": "A shiny, colorful paradise on a floating island, in the middle of dark, horrifying void. Epic Realistic, masterpiece, (hdr:1.4), (intricate details, hyperdetailed:1.2), artstation, vignette, complex background",
    "oasis": "A beautiful oasis in the middle of the endless void. Epic Realistic, (hdr:1.4), (intricate details, hyperdetailed:1.2), artstation, vignette, complex background",
    "robrob": "A futuristic factory, with robots building robots. Epic Realistic, (hdr:1.4), (intricate details, hyperdetailed:1.2), artstation, vignette, complex background",
    "lion": "A majestic lion jumping from a big stone at night, with star-filled skies. Hyperdetailed, with Complex tropic, African background.",
}

promptneg={
    "lion": "extra limbs",
}


def get_prompt(promptname):
    if promptname in promptdict:
        prompt = promptdict[promptname]
    else:
        prompt = promptname
    
    if promptname in promptneg:
        neg = promptneg[promptname]
    else:
        neg = "blurry"

    return prompt, neg
