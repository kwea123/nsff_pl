import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def visualize_mask(mask, cmap=cv2.COLORMAP_BONE):
    """
    mask: (H, W) in 0~1
    """
    x = mask.cpu().numpy()
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def blend_images(img1, img2, alpha):
    """
    alpha blend two images: img1 * alpha + img2 * (1-alpha)
    img1 and img2: (3, H, W)
    """
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img1 = (255*img1).astype(np.uint8)
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    img2 = (255*img2).astype(np.uint8)
    blend = cv2.addWeighted(img1, alpha, img2, 1-alpha, 2.2)
    x_ = Image.fromarray(blend)
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_