import torch

from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import os
import numpy as np
import torch
from torchvision.models import inception_v3
from scipy.stats import entropy
from torchvision import transforms


def from_dir(directory):

    images = []
    paths = os.listdir(directory)

    for path in paths:
        if path.endswith(".png"):
            img = Image.open(os.path.join(directory, path))
            images.append(img)

    return images

def clip_eval(images, prompt):

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

    N = len(images)
    H = images[0].height
    W = images[0].width
    C = 3

    np_images = np.zeros((N, H, W, C))
    for i in range(len(images)):
        img = np.array(images[i])
        H = img.shape[0]
        W = img.shape[1]
        np_images[i] = img

    prompts = [prompt] * N

    score = clip_score_fn(torch.from_numpy(np_images).permute(0, 3, 1, 2).cuda(), prompts).detach()
    return round(float(score), 4)

def clip_eval_std(images, prompt,splits=4, type = "base"):
    N = len(images)


    scores = np.zeros(splits)
    for i in range(splits):
        start = i * (N // splits)
        end = (i + 1) * (N // splits)

        if type == "base":
            scores[i] = clip_eval(images[start:end], prompt)
        elif type == "large":
            scores[i] = clip_eval_large(images[start:end], prompt)

    return np.mean(scores), np.std(scores)




def clip_eval_large(images, prompt):

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14-336")

    N = len(images)
    H = images[0].height
    W = images[0].width
    C = 3
    np_images = np.zeros((N, H, W, C))




    for i in range(len(images)):
        img = np.array(images[i])
        H = img.shape[0]
        W = img.shape[1]
        np_images[i] = img

    prompts = [prompt] * N

    score = clip_score_fn(torch.from_numpy(np_images).permute(0, 3, 1, 2), prompts).detach()

    return round(float(score), 4)


def inception_score(images, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs"""

    N = len(images)
    H = images[0].height
    W = images[0].width
    C = 3
    imgs = torch.zeros((N, C, H, W))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for i in range(len(images)):
        img = transform(images[i])
        imgs[i] = img

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    dtype = torch.cuda.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)