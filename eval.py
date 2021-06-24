from argparse import ArgumentParser
import cv2
import copy
from collections import defaultdict
import imageio
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm

from models.rendering import render_rays, interpolate
from models.nerf import PosEmbedding, NeRF

from utils import load_ckpt, visualize_depth
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
from datasets.ray_utils import *

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='monocular',
                        choices=['monocular'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='''test or test_spiral or 
                                test_spiralX or test_fixviewX_interpY''')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 288],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--start_end', nargs="+", type=int, default=[0, 100],
                        help='start frame and end frame')

    parser.add_argument('--use_viewdir', default=False, action="store_true",
                        help='whether to use view dependency in static network')
    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=48,
                        help='number of embeddings for transient objects')
    parser.add_argument('--flow_scale', type=float, default=0.2,
                        help='flow scale to multiply to flow network output')
    parser.add_argument('--output_transient', default=False, action="store_true",
                        help='whether to output the full result (static+transient)')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['mp4', 'gif'],
                        help='which format to save')
    parser.add_argument('--fps', type=int, default=10,
                        help='video frame per second')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='png',
                        help='which format to save')

    parser.add_argument('--save_error', default=False, action="store_true",
                        help='whether to save error map')

    return parser.parse_args()


@torch.no_grad()
def f(models, embeddings,
      rays, ts, max_t, N_samples, N_importance,
      chunk,
      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    kwargs_ = copy.deepcopy(kwargs)
    for i in range(0, B, chunk):
        if 'view_dir' in kwargs:
            kwargs_['view_dir'] = kwargs['view_dir'][i:i+chunk]
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        None if ts is None else ts[i:i+chunk],
                        max_t,
                        N_samples,
                        0,
                        0,
                        N_importance,
                        chunk,
                        test_time=True,
                        **kwargs_)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def save_depth(depth, h, w, dir_name, filename):
    depth_pred = np.nan_to_num(depth.view(h, w).numpy())
    depth_pred_img = visualize_depth(torch.from_numpy(depth_pred)).permute(1, 2, 0).numpy()
    depth_pred_img = (depth_pred_img*255).astype(np.uint8)
    imageio.imwrite(os.path.join(dir_name, filename), depth_pred_img)
    return depth_pred_img


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': (w, h),
              'start_end': tuple(args.start_end)}
    dataset = dataset_dict[args.dataset_name](**kwargs)

    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {'K': dataset.K, 'dataset': dataset}

    if args.split.startswith('test_fixview') and int(args.split.split('_')[-1][6:])>0:
        kwargs['output_transient'] = True
        kwargs['output_transient_flow'] = ['fw', 'bw']
    else:
        kwargs['output_transient'] = args.output_transient
        kwargs['output_transient_flow'] = []

    embeddings = {'xyz': PosEmbedding(9, 10), 'dir': PosEmbedding(3, 4)}

    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).to(device)
        embeddings['a'] = embedding_a
        load_ckpt(embedding_a, args.ckpt_path, 'embedding_a')
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).to(device)
        embeddings['t'] = embedding_t
        load_ckpt(embedding_t, args.ckpt_path, 'embedding_t')

    nerf_fine = NeRF(typ='fine',
                     use_viewdir=args.use_viewdir,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     output_flow=len(kwargs['output_transient_flow'])>0,
                     flow_scale=args.flow_scale).to(device)
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    models = {'fine': nerf_fine}
    if args.N_importance > 0:
        raise ValueError("coarse to fine is not ready now! please set N_importance to 0!")
    #     nerf_coarse = NeRF(typ='coarse',
    #                        use_viewdir=args.use_viewdir,
    #                        encode_transient=args.encode_t,
    #                        in_channels_t=args.N_tau).to(device)
    #     load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    #     models['coarse'] = nerf_coarse

    imgs, depths, psnrs = [], [], []

    last_results = None
    for i in tqdm(range(len(dataset))):
        if args.split.startswith('test_fixview') and i==len(dataset)-1: # last frame
            img_pred = last_results['rgb_fine'].view(h, w, 3).numpy()
            img_pred = (255*np.clip(img_pred, 0, 1)).astype(np.uint8)
            imgs += [img_pred]
            imageio.imwrite(os.path.join(dir_name, f'{i:03d}_{int(0):03d}.png'), img_pred)
            if args.save_depth:
                depths += [save_depth(last_results['depth_fine'], h, w,
                                      dir_name, f'depth_{i:03d}_{int(0):03d}.png')]
        else:
            sample = dataset[i]
            if args.split.startswith('test_spiral') and 'view_dir' not in kwargs:
                kwargs['view_dir'] = dataset[0]['rays'][:, 3:6].to(device)
            ts = None if 'ts' not in sample else sample['ts'].to(device)
            if last_results is None:
                results = f(models, embeddings, sample['rays'].to(device), ts,
                            dataset.N_frames-1, args.N_samples, args.N_importance,
                            args.chunk, **kwargs)
            else: results = last_results

            if args.split.startswith('test_fixview'):
                interp = int(args.split.split('_')[-1][6:])
                results_tp1 = f(models, embeddings, sample['rays'].to(device), ts+1,
                                dataset.N_frames-1, args.N_samples, args.N_importance,
                                args.chunk, **kwargs)
                for dt in np.linspace(0, 1, interp+1)[:-1]: # interp images
                    if dt == 0:
                        img_pred = results['rgb_fine'].view(h, w, 3)
                        depth_pred = results['depth_fine']
                    else:
                        img_pred, depth_pred = interpolate(results, results_tp1, 
                                        dt, dataset.Ks[sample['cam_ids']], sample['c2w'], (w, h))
                    img_pred = (255*np.clip(img_pred.numpy(), 0, 1)).astype(np.uint8)
                    imgs += [img_pred]
                    imageio.imwrite(os.path.join(dir_name, f'{i:03d}_{int(dt*100):03d}.png'), img_pred)
                    if args.save_depth:
                        depths += [save_depth(depth_pred, h, w,
                                              dir_name, f'depth_{i:03d}_{int(dt*100):03d}.png')]
                last_results = results_tp1
            else: # one image
                img_pred = np.clip(results['rgb_fine'].view(h, w, 3).numpy(), 0, 1)
                img_pred_ = (img_pred*255).astype(np.uint8)
                imgs += [img_pred_]
                imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
                if args.save_depth:
                    depths += [save_depth(results['depth_fine'], h, w,
                                          dir_name, f'depth_{i:03d}.png')]

        if args.split == 'test':
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

            if args.save_error:
                err = torch.abs(img_gt-img_pred).sum(-1).numpy()
                err = (err-err.min())/(err.max()-err.min())
                err_vis = cv2.applyColorMap((err*255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(dir_name, f'err_{i:03d}.png'), err_vis)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
        np.save(os.path.join(dir_name, 'psnr.npy'), np.array(psnrs))

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                    imgs, fps=args.fps)
    if args.save_depth:
        imageio.mimsave(os.path.join(dir_name, f'depth_{args.scene_name}.{args.video_format}'),
                        depths, fps=args.fps)