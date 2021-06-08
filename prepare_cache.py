import argparse
from datasets import dataset_dict
import os
import torch

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='lightfield',
                        choices=['lightfield', 'distortion'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--use_distorted_images', default=False, action="store_true",
                        help='Train using distorted images or not, used when dataset_name==distortion')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    dataset = dataset_dict[args.dataset_name]
    kwargs = {'root_dir': args.root_dir,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'lightfield':
        kwargs['cam_train'] = [-1]
    elif args.dataset_name == 'distortion':
        kwargs['use_distorted_images'] = args.use_distorted_images
    train_dataset = dataset(split='train', **kwargs)

    if args.dataset_name == 'lightfield':
        cache_dir = f'cache_{args.dataset_name}_{args.img_wh[0]}_{args.img_wh[1]}_camall'
    elif args.dataset_name == 'distortion':
        if kwargs['use_distorted_images']:
            cache_dir = f'cache_{args.dataset_name}_{args.img_wh[0]}_{args.img_wh[1]}_use_distorted_images'
        else:
            cache_dir = f'cache_{args.dataset_name}_{args.img_wh[0]}_{args.img_wh[1]}_use_undistorted_images'

    cache_dir = os.path.join(args.root_dir, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    torch.save(train_dataset.rays_dict, os.path.join(cache_dir, 'rays_dict.pt'))
    torch.save(train_dataset.rgbs_dict, os.path.join(cache_dir, 'rgbs_dict.pt'))
    print(f'Cache saved to {cache_dir} !')