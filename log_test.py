import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import tqdm


def load_stack(filepath, channel=0, n_channels=1):

    with tiff.TiffFile(filepath, is_ome=False, is_mmstack=False) as tif:
        pages = tif.pages[channel::n_channels]
        raw_stack = np.stack([p.asarray() for p in pages], axis=0)

    p_low, p_high = np.percentile(raw_stack[:, ::4, ::4], (1, 99.8))

    vals = np.arange(65536, dtype=np.float32)
    lut = np.clip((vals - p_low) / (p_high - p_low + 1e-8) * 255, 0, 255).astype(np.uint8)

    return lut[raw_stack]



def laplacian_of_gaussian (img, background) : 

    diff = np.abs(img.astype(np.float32) - background).astype(np.uint8)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.0)

    log = cv2.Laplacian(blurred, ddepth=cv2.CV_16S, ksize=3)
    return img, log



def visualize_debug_stages(img, log, log_vis, zero_crossings):
    _, axes = plt.subplots(2, 1, figsize=(14, 14))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"Image de base")
    axes[0, 1].imshow(log, cmap="gray")
    axes[0, 1].set_title("Laplacian of gaussian après background substraction")
    axes[1, 0].imshow(log_vis, cmap="gray")
    axes[1, 0].set_title("Laplacian of gaussian + 8bit(??)")
    axes[1, 1].imshow(zero_crossings, cmap="gray")
    axes[1, 1].set_title("zero crossings")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"
    print("debut")
    stack = load_stack(tif_path, channel=0, n_channels=1)
    print("stack loaded")
    background = np.median(stack, axis=0).astype(np.float32)
    print("background)")
    img, log = laplacian_of_gaussian(stack[0], background)
    print("log")
    visualize_debug_stages(img, log)



