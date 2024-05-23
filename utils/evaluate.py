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
import cv2
from scipy import stats
from transformers import AutoProcessor, AutoModel
from pandas import DataFrame
from tqdm import tqdm
from pytorch_fid import fid_score
from torchvision.transforms import ToTensor
import pandas as pd
from scipy.fftpack import dct, idct
from joblib import Memory
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers import DiffusionPipeline
from skimage.metrics import structural_similarity
from pytorch_msssim import ssim

memory = Memory(location="/tmp/mem2", verbose=0)
##memory = Memory(location=None, verbose=0)


def from_dir(directory):

    images = []
    paths = os.listdir(directory)

    for path in paths:
        if path.endswith(".png"):
            img = Image.open(os.path.join(directory, path))
            images.append(img)

    return images

def from_dirs(directories, idxes, type = "png",img_idx = None):
    images = []
    if type == "jpeg" or type == "jpev_cv2":
        ##  fname = os.path.join(subfolder, "%d.jpeg" % img_idx[idx])
        img_names = ["%d.jpeg" % img_idx[idx] for idx in idxes]
    else:
        img_names = ["img_%05d.png" % idx for idx in idxes]

    for directory in directories:
        for img_name in img_names:
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                if type == "png":
                    img = Image.open(img_path)
                elif type == "jpeg":
                    img = Image.open(img_path)
                elif type == "cv2" or type == "jpeg_cv2":
                    img = cv2.imread(img_path)
                elif type == "torch":
                    img = ToTensor()(Image.open(img_path))
                elif type == "np":
                    img = np.array(Image.open(img_path))
                images.append(img)
            else:
                print(f"Image {img_name} not found in {directory}")
    return images

def dct_fn(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def diff_eval(baseline, images, fn = None):

    if isinstance(baseline, Image.Image):
        cast_fn = lambda x: np.array(x)
    elif isinstance(baseline, np.ndarray):
        cast_fn = lambda x: x
    elif isinstance(baseline, torch.Tensor):
        cast_fn = lambda x: x.cpu().numpy()

    baseline_np = cast_fn(baseline)
    
    ##baseline_np = fn(baseline_np)
    ##mses = [(np.mean((baseline_np - fn(cast_fn(img))) ** 2)) for img in images]
    mses = [fn(baseline_np, cast_fn(img)) for img in images]

    return mses






def get_fn_from_desc(fn_desc):
    if fn_desc is None or fn_desc == "pmse":
        fn = lambda x: x


        fn = lambda x,y: np.mean((x - y) ** 2)
    elif fn_desc == "dct" or fn_desc == "fmse":
        fn1 = lambda x: dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')/10
        fn = lambda x,y: np.mean((fn1(x) - fn1(y)) ** 2)
    elif fn_desc == "lfmse":
        k = 32
        K = 1024//32
        fn1 = lambda x: dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')[:k,:k,:]/(K*10)
        fn = lambda x,y: np.mean((fn1(x) - fn1(y)) ** 2)

    elif fn_desc == "latent" or fn_desc == "lmse":
        name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        base = DiffusionPipeline.from_pretrained(
            name_or_path, torch_dtype=torch.float32, use_safetensors=True,
        )
        vae = base.vae.cuda()

        ##transform int8 image to fp32:

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        fn1 = lambda x: retrieve_latents(vae.encode(transform(x).cuda().unsqueeze(0)))
        fn = lambda x,y: torch.mean((fn1(x) - fn1(y)) ** 2).detach().cpu().numpy()

    elif fn_desc == "ssim":
        ##fn = lambda x,y: 1-compare_ssim(x.transpose((1, 2, 0)), y.transpose((1, 2, 0)))
        fn = lambda x,y: structural_similarity(x,y,channel_axis = 2)
    elif fn_desc == "ssim+":
        transform = transforms.ToTensor()
        fn = lambda x,y: ssim(transform(x).unsqueeze(0).cuda(), transform(y).unsqueeze(0).cuda(), data_range=1.0).detach().cpu().numpy()
    elif fn_desc == "psnr":
        fn = lambda x,y: cv2.PSNR(x, y)
    
    return fn

@memory.cache
def eval_mse(runs = None, idxes = None, baseline = 0, fn_desc = None, is_jpeg = False):
        
        ## must have either images and cols or runs and idxes
        images = from_dirs(runs, idxes, type = 'png' if not is_jpeg else 'jpeg')
        cols = len(idxes)

        return eval_mse_imgs(images, cols, baseline, fn_desc)

def eval_mse_imgs(images = None , cols = None, baseline = 0, fn_desc = None):
        
        ## must have either images and cols or runs and idxes
        
        fn = get_fn_from_desc(fn_desc)

        samples = len(images)
        
        rows = samples // cols
        mses = np.zeros((cols, rows))
        for c in range(cols):
            column_idx = range(c, samples, cols)
            column = [images[i] for i in column_idx]
            mses[c,:] = diff_eval(column[baseline], column, fn = fn)

        return mses

def eval_mse_stats(runs, idxes, baseline = 0, fn_desc = None):

    if isinstance(idxes, int):
        idxes = list(range(idxes))

    mses = eval_mse(runs = runs, idxes = idxes, baseline = baseline, fn_desc = fn_desc)
    mean = mses.mean(axis=0)
    std = mses.std(axis=0)

    return mean, std

def put_in_memory(baseline, run, idx, fn_desc, value):


    if baseline[-1] == "/":
        baseline = baseline[:-1]
    if run[-1] == "/":
        run = run[:-1]

    mem_dir = "objects"

    if not os.path.exists(mem_dir):
        os.makedirs(mem_dir)

    baseline_name = baseline.split("/")[-1]
    mem_path = f"{mem_dir}/{baseline_name}_{fn_desc}.obj"

    if os.path.exists(mem_path):
        with open(mem_path, "rb") as f:
            memory = torch.load(f)
    else:
        memory = {}

    key = f"{run.split('/')[-1]}_{idx}"

    memory[key] = value
    return

def get_from_memory(baseline, run, idx, fn_desc):

    if baseline[-1] == "/":
        baseline = baseline[:-1]
    if run[-1] == "/":
        run = run[:-1]

    mem_dir = "objects"

    if not os.path.exists(mem_dir):
        os.makedirs(mem_dir)

    baseline_name = baseline.split("/")[-1]
    mem_path = f"{mem_dir}/{baseline_name}_{fn_desc}.obj"

    if os.path.exists(mem_path):
        with open(mem_path, "rb") as f:
            memory = torch.load(f)
    else:
        memory = {}

    key = f"{run.split('/')[-1]}_{idx}"

    if key in memory:
        return memory[key]
    else:
        return None


@memory.cache
def eval_mse_matrix(runs, idxes, fn_desc = None, stop = False):

    if isinstance(idxes, int):
        idxes = list(range(idxes))

    format = "png" 

    if "coco" in runs[0]:
        from T2IBenchmark.datasets import get_coco_30k_captions
        format = "jpeg"
        img_idx = list(get_coco_30k_captions().keys())
    elif "sdx" in runs[0]:
        format = "jpeg"
        img_idx = idxes

    images = from_dirs(runs, idxes, type = format, img_idx = img_idx)


    cols = len(idxes)
    rows = len(runs)

    matrix_means = np.zeros((rows, rows))
    matrix_stds = np.zeros((rows, rows))

    for baseline in range(rows):
        mses = eval_mse_imgs(images = images, cols = cols, baseline = baseline, fn_desc = fn_desc)
        mean = mses.mean(axis=0)
        std = mses.std(axis=0)
        matrix_means[baseline,:] = mean
        matrix_stds[baseline,:] = std
        if stop:
            matrix_means = matrix_means[0:1,:]
            matrix_stds = matrix_stds[0:1,:]
            break
            
    return matrix_means, matrix_stds

@memory.cache
def eval_mse_pval(baseline_run, sample1_run, sample2_run, idxes, fn_desc = None):

    if isinstance(idxes, int):
        idxes = list(range(idxes))

    runs = [baseline_run, sample1_run, sample2_run]
    format = "png" 

    if "coco" in runs[0]:
        from T2IBenchmark.datasets import get_coco_30k_captions
        format = "jpeg"
        img_idx = list(get_coco_30k_captions().keys())
    elif "sdx" in runs[0]:
        format = "jpeg"
        img_idx = idxes

    images = from_dirs(runs, idxes, type = format, img_idx = img_idx)
    
    cols = len(idxes)

    mses = eval_mse_imgs(images = images, cols = cols, baseline = 0, fn_desc = fn_desc)
    sample1 = mses[:,1]
    sample2 = mses[:,2]

    t, p_value_two_sided = stats.ttest_ind(sample1, sample2)

    if t>0:
        p_value_one_sided = p_value_two_sided/2
    else:
        p_value_one_sided = 1 - p_value_two_sided/2

    return p_value_one_sided

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
    assert N >= batch_size

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

    if splits > 1:
        return np.mean(split_scores), np.std(split_scores)
    else:
        return np.mean(split_scores), 0

def make_tmp_folder(idx = 0):
    name = f"/tmp/eval{idx}"
    if os.path.exists(name):
        os.system(f"rm -rf {name}")

    os.makedirs(name, exist_ok=True)
    return name

def remove_tmp_folder(idx = 0):
    name = f"/tmp/eval{idx}"
    os.system(f"rm -rf {name}")


def calculate_fid(baseline_dir, dirs, device="cuda", dim=2048, max = -1, batch_size=16):
    # Calculate the FID scores
    results = []

    if max > 0:
        images = from_dir(baseline_dir)[:max]
        baseline_images = make_tmp_folder(0)
        for i, img in enumerate(images):
            img.save(f"{baseline_images}/{i}.png")
    else:
        baseline_images = baseline_dir

    for dir in tqdm(dirs, total=len(dirs), desc="Calculating FID"):
        if max > 0:
            images = from_dir(dir)[:max]
            folder = make_tmp_folder(1)
            for i, img in enumerate(images):
                img.save(f"{folder}/{i}.png")
        else:
            folder = dir
        # Calculate the FID score
        fid = fid_score.calculate_fid_given_paths([baseline_images, folder], batch_size, device, dim, num_workers=0)
        results.append(fid)
        
    if max > 0:
        remove_tmp_folder(0)
        remove_tmp_folder(1)

    results = np.array(results).reshape(1, -1)
    # Create a DataFrame with the results
    rows = ["FID"]
    cols = [dir.split("/")[-1] for dir in dirs]
    df = pd.DataFrame(results, index=rows, columns=cols)

    return df


def eval_clip_tbl(runs, prompt,max=-1):


    ## need a matrix of strings in size 2 x len(runs)
    rows = ["clip", "clip_large"] 
    cols = [run.split("/")[-1] for run in runs]

    df = DataFrame("", index=["clip", "clip_large"], columns=[run.split("/")[-1] for run in runs])

    for i,run in tqdm(enumerate(runs), total=len(runs)):

        images = from_dir(run)
        if max > 0:
            images = images[:max]
        mean,std = clip_eval_std(images,prompt,splits=4,type="base")
        df.iloc[0,i] = f"{mean:.4f} ± {std:.4f}"
        mean,std = clip_eval_std(images,prompt,splits=4,type="large")
        df.iloc[1,i] = f"{mean:.4f} ± {std:.4f}"

    torch.cuda.empty_cache()


    return df

def eval_pickapick_tbl(runs, prompt, device="cuda", batch_size=4):

    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    results = []

    for run in runs:
        mean, std = eval_pickapick(run, prompt, device, batch_size, processor, model)
        res = f"{mean:.4f} ± {std:.4f}"
        results.append(res)

    results = np.array(results).reshape(1, -1)

    rows = ["pickapick"] 
    cols = [run.split("/")[-1] for run in runs]
    df = DataFrame(results, index=rows, columns=cols)
    return df




def eval_pickapick(directory, prompt, device="cuda", batch_size=4, processor = None, model = None):
    images = from_dir(directory)

    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


    if processor is None:
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
    if model is None:
        model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    scores = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]

            # preprocess
            image_inputs = processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            batch_scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            scores.append(batch_scores)

        # Concatenate all scores
        scores = torch.cat(scores, dim=0)

        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1).cpu()

    mean = probs.mean().item()
    std = probs.std().item()

    return mean, std