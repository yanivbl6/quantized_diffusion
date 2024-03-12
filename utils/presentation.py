import matplotlib.pyplot as plt
import math
import os 
from PIL import Image, ImageDraw, ImageFont
import wandb
import numpy as np

from contextlib import contextmanager
from matplotlib.backends.backend_pdf import PdfPages
from utils.evaluate import from_dirs, eval_mse, inception_score
import cv2

@contextmanager
def suppress_plot_show():
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


def print_matrices(mean, std, row_names=None):
    np.set_printoptions(precision=2)

    # Ensure that mean and std have the same shape
    assert mean.shape == std.shape, "Mean and std must have the same shape"

    idx_min = (mean + np.diag([np.inf]* mean.shape[0])).argmin(axis=0)

    # Print column names if provided
    if row_names is not None:
        print("\t\t" + "\t".join([name.ljust(15) for name in row_names]))

    # Iterate over the elements of the matrices
    for i in range(mean.shape[0]):
        # Print row name if provided
        if row_names is not None:
            print(row_names[i].ljust(15), end="\t")

        for j in range(mean.shape[1]):
            # Check if the current value is the smallest in its row (excluding the diagonal)
            if j == idx_min[i]:
                # If it is, print it in bold
                print(f"\033[1;31m{mean[i, j]:.2f} ± {std[i, j]:.2f}\033[0m", end="\t")
            else:
                # Otherwise, print it normally
                if i==j:
                    print(f"------------", end="\t")
                else:
                    print(f"{mean[i, j]:.2f} ± {std[i, j]:.2f}", end="\t")
        print()  # Print a newline at the end of each row


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
    if samples > 64:
        images = images[:64]
        samples = 64


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




def plot_hybrid_grid(name, runs, idxes=0,directory = "images", per_page = None, show_mse = False):

    ## if runs is None, use all directories in the directory, but only directories
    if runs is None:
        runs = os.listdir(directory)
        runs = [os.path.join(directory, run) for run in runs if os.path.isdir(os.path.join(directory, run))]

    if isinstance(idxes, int):
        idxes = list(range(idxes))
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
    img_size  = 40//maxnm

    per_page = max(m,5) if per_page is None else per_page
    if per_page > n:
        per_page = n

    num_pages = (n+per_page-1) // per_page
    
    cols = m

    fname = directory + "/" + name + ".pdf"


    if show_mse:
        mses = eval_mse(images, cols)
        
    with PdfPages(fname) as pdf:
        for p in range(num_pages):

            rows = min(per_page, n - p*per_page)
            fig, ax = plt.subplots(per_page, cols, figsize=(per_page*img_size, cols*img_size))

            for r in range(rows):
                i = p*per_page + r
                title = runs[i].split("/")[-1]

                ax[r,0].set_title(title, fontsize=26)
                for j in range(cols):
                    if i*m+j < samples:
                        ##images are Image objects
                        img = images[i*m+j]

                        if show_mse:
                            draw = ImageDraw.Draw(img)
                            text = f"MSE: {mses[j,i]:.4f}"
                            text_position = (50, 50)  # Adjust as needed
                            font = ImageFont.load_default()
                            font = font.font_variant(size=50)
                            draw.text(text_position, text, fill="white", font=font)


                        ax[r, j].imshow(img)
                        ax[r, j].axis('off')
                    else:
                        ## invisible image
                        ax[r, j].imshow(np.zeros((10,10,3)))
                        ax[r, j].axis('off')
                        

            plt.suptitle(name + f", page {p+1}/{num_pages}", fontsize=50)
            pdf.savefig(fig)


    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            plt.show()
    except NameError:
        pass

    plt.close()

def create_video(name, runs, idxes, directory="images", seconds = 3, 
                 display_mse = False, display_is = True, repeat_baseline = False):


    if isinstance(idxes, int):
        idxes = list(range(idxes))

    fps = 1
    # if runs is None, use all directories in the directory, but only directories
    if runs is None:
        runs = os.listdir(directory)
        runs = [os.path.join(directory, run) for run in runs if os.path.isdir(os.path.join(directory, run))]

    if isinstance(runs, str):
        runs = [runs]

    # check that all runs are valid subdirectories
    for i in range(len(runs)):
        if not os.path.exists(runs[i]):
            runs[i] = os.path.join(directory, runs[i])    
        assert os.path.exists(runs[i]), "Invalid run directory: " + runs[i]



    images = from_dirs(runs, [idxes[0]], type = "cv2")

    if display_mse:
        mses = eval_mse(runs=runs, idxes=idxes, baseline=0)



    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_name = directory + "/" + name + ".mp4"
    height, width, _ = images[0].shape
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for j,idx in enumerate(idxes):

        if display_is:
            images_png = from_dirs(runs, [idx], type = "png")
            is_scores = [inception_score([img],1)[0] - 1.0 for img in images_png]
            del images_png
        
        images = from_dirs(runs, [idx], type = "cv2")




        if repeat_baseline:
            image_indxes = []
            for i in range(1,len(images)):
                image_indxes.append(0)
                image_indxes.append(i)
        else:
            image_indxes = range(len(images))

        for i in image_indxes:
            # Convert the image from BGR to RGB color format (This step is needed if the image is read using cv2.imread())
            
            image = images[i]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            title = runs[i].split("/")[-1]
            cv2.putText(image, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if display_mse:
                mse = mses[j,i]
                cv2.putText(image, f"MSE: {mse:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if display_is:
                is_score = is_scores[i]
                cv2.putText(image, f"IS: {is_score:.5f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            for _ in range(fps*seconds):
                video.write(image)

    video.release()

    print(f"Video saved under {video_name}")


class WandbRuns:
    _runs = None

    @classmethod
    def get_runs(cls, path):
        if cls._runs is None:
            api = wandb.Api()
            cls._runs = api.runs(path)
        return cls._runs


def plot_lines(path = "dl-projects/qpipe_scores", baseline = False, 
                count = 256, dtype = "M2E5", filter = lambda x: True,
                label = "", field = "clip_score_mean", verbose = False):
    


    runs = WandbRuns.get_runs(path)

    names = [run.name for run in runs]

    mruns = []
    for run in runs:

        if baseline:
            if "M23E8" in run.name: 
                if run.name == f"A_M23E8_W_{dtype}_flex_V3":
                    y_bl = run.summary[field]
                    dy_bl = run.summary[field + "_std"]

        if not 'count' in run.summary or run.summary['count'] != count:
            continue
                
        if dtype in run.name:
            if filter(run.name):
                mruns.append(run)
                if verbose:
                    print(run.name)



    y = np.zeros(len(mruns))
    dy = np.zeros(len(mruns))

    x = np.zeros(len(mruns))

    for i, run in enumerate(mruns):
        df = run.summary
        y[i] = df[field]
        dy[i] = df[field + "_std"]
        x[i] = run.config['num_inference_steps']


    ## sory x, y, dy according to x:
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    dy = dy[idx]

    plt.fill_between(x, y-dy, y+dy, alpha=0.6, label=label)
    plt.plot(x, y)

    if baseline:
        y_bl = np.asarray([y_bl] * len(x))
        dy_bl = np.asarray([dy_bl] * len(x))
        plt.fill_between(x, y_bl-dy_bl, y_bl+dy_bl, color='k', alpha=0.2, label="baseline")
        plt.plot(x, y_bl, color='k')

