import matplotlib.pyplot as plt
import math
import os 
from PIL import Image

def from_dir(directory):

    images = []
    paths = os.listdir(directory)

    for path in paths:
        if path.endswith(".png"):
            img = Image.open(os.path.join(directory, path))
            images.append(img)

    return images


def plot_grid(name, images, directory = "images", title = None):
    samples = len(images)

    n = math.ceil(math.sqrt(samples))
    m = math.ceil(samples/n)

    maxnm = max(n,m)
    img_size  = 30//maxnm
    fig, ax = plt.subplots(n, m, figsize=(m*img_size, n*img_size))
    for i in range(n):
        for j in range(m):
            if i*m+j < samples:

                ##images are Image objects
                img = images[i*m+j]
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
    
    if title is not None:
        fig.suptitle(title, fontsize=50)
    else:
        fig.suptitle(name.split("/")[-1], fontsize=50)

    plt.savefig(directory + "/" + name + ".pdf", format= 'pdf')

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            plt.show()
    except NameError:
        pass

    plt.close()


def from_dirs(directories, idxes):
    images = []
    img_names = ["img_%05d.png" % idx for idx in idxes]

    for directory in directories:
        for img_name in img_names:
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)

    return images

def plot_hybrid_grid(name, runs, idxes=0,directory = "images"):

    ## if runs is None, use all directories in the directory, but only directories
    if runs is None:
        runs = os.listdir(directory)
        runs = [os.path.join(directory, run) for run in runs if os.path.isdir(os.path.join(directory, run))]

    if isinstance(idxes, int):
        idxes = [idxes]
    if isinstance(runs, str):
        runs = [runs]



    ## check that all runs are valid subdirectories
    for i in range(len(runs)):
        if not os.path.exists(runs[i]):
            runs[i] = os.path.join(directory, runs[i])    
        assert os.path.exists(runs[i]), "Invalid run directory: " + runs[i]

    images = from_dirs(runs, idxes)

    samples = len(images)

    n = len(runs)
    m = len(idxes)

    maxnm = max(n,m)
    img_size  = 30//maxnm
    fig, ax = plt.subplots(n, m, figsize=(m*img_size, n*img_size))
    for i in range(n):
        title = runs[i].split("/")[-1]
        ax[i,0].set_title(title, fontsize=26)
        for j in range(m):
            if i*m+j < samples:

                ##images are Image objects
                img = images[i*m+j]
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                
    plt.suptitle(name, fontsize=50)
    plt.savefig(directory + "/" + name + ".pdf", format= 'pdf')

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            plt.show()
    except NameError:
        pass

    plt.close()
