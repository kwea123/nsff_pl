import torch
from einops import rearrange
from kornia.losses import ssim_loss


def mse(image_gt, image_pred, valid_mask=None, reduction='mean'):
    value = (image_gt-image_pred)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_gt, image_pred, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_gt, image_pred, valid_mask, reduction))


def ssim(image_gt, image_pred, valid_mask=None, window_size=11, reduction='mean'):
    """
    image_pred and image_gt: (H, W, 3)
    valid_mask: (H, W)
    """
    value = ssim_loss(rearrange(image_gt, 'h w c -> 1 c h w'),
                      rearrange(image_pred, 'h w c -> 1 c h w'),
                      window_size=window_size, reduction='none')
    value = rearrange(value, '1 c h w -> h w c')
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return 1-torch.mean(value)
    return 1-value


@torch.no_grad()
def lpips(lpips_model, image_gt, image_pred, valid_mask=None, reduction='mean'):
    """
    lpips_model: alexnet.
    image_pred and image_gt: (H, W, 3) in [0, 1]
    valid_mask: (H, W)
    """
    value = lpips_model(image_gt.permute(2, 0, 1).unsqueeze(0),
                        image_pred.permute(2, 0, 1).unsqueeze(0),
                        normalize=True).squeeze()
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        value = value.mean()
    return value