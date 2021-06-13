import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2

def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data for nsff training')
    parser.add_argument('--root_dir', type=str, help='data root directory', required=True)
    parser.add_argument('--cuda-device',type=str,default='0',help='cuda device to use')

    parser.add_argument('--max-width', type=int, default=1280, help='max image width')
    parser.add_argument('--max-height', type=int, default=720, help='max image height')
    parser.add_argument(
        '--images-resized', default='images_resized', help='location for resized/renamed images')
    parser.add_argument('--image_input', default='frames', help='location for original images')
    parser.add_argument(
        '--undistorted-output', default='images', help='location of undistorted images')
    parser.add_argument(
        '--overwrite', default=False,action='store_true', help='overwrite cache')

    args = parser.parse_args()
    return args

def resize_frames(args):
    vid_name = os.path.basename(args.root_dir)
    frames_dir = os.path.join(args.root_dir, args.images_resized)
    os.makedirs(frames_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.root_dir, args.image_input, '*.jpg')) +
        glob.glob(os.path.join(args.root_dir, args.image_input, '*.png')))

    print('Resizing images ...')
    for file_ind, file in enumerate(tqdm(files, desc=f'imresize: {vid_name}')):
        out_frame_fn = f'{frames_dir}/{file_ind:05}.png'

        # skip if both the output frame and the mask exist
        if os.path.exists(out_frame_fn) and not args.overwrite:
            continue

        im = cv2.imread(file)

        # resize if too big
        if im.shape[1] > args.max_width or im.shape[0] > args.max_height:
            factor = max(im.shape[1] / args.max_width, im.shape[0] / args.max_height)
            dsize = (int(im.shape[1] / factor), int(im.shape[0] / factor))
            im = cv2.resize(src=im, dsize=dsize, interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_frame_fn, im)

def generate_masks(args):
    # ugly hack, masks expects images in images, but undistorted ones are going there later
    undist_dir = os.path.join(args.root_dir, args.undistorted_output)
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)
        os.system(f'cp -r {args.root_dir}/{args.images_resized}/*.png {args.root_dir}/images')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python third_party/predict_mask.py --root_dir {args.root_dir}')
        os.system(f'rm {args.root_dir}/images')

def run_colmap(args):
    max_num_matches = 132768  # colmap setting

    if not os.path.exists(f'{args.root_dir}/database.db') or args.overwrite:
        os.system(f'''
            CUDA_VISIBLE_DEVICES={args.cuda_device} colmap feature_extractor \
                --database_path={args.root_dir}/database.db \
                --image_path={args.root_dir}/{args.images_resized}\
                --ImageReader.mask_path={args.root_dir}/masks \
                --ImageReader.camera_model=SIMPLE_RADIAL \
                --ImageReader.single_camera=1 \
                --ImageReader.default_focal_length_factor=0.95 \
                --SiftExtraction.peak_threshold=0.004 \
                --SiftExtraction.max_num_features=8192 \
                --SiftExtraction.edge_threshold=16''')

        os.system(f'''
            CUDA_VISIBLE_DEVICES={args.cuda_device} colmap exhaustive_matcher \
                --database_path={args.root_dir}/database.db \
                --SiftMatching.multiple_models=1 \
                --SiftMatching.max_ratio=0.8 \
                --SiftMatching.max_error=4.0 \
                --SiftMatching.max_distance=0.7 \
                --SiftMatching.max_num_matches={max_num_matches}''')

    if not os.path.exists(f'{args.root_dir}/sparse') or args.overwrite:
        os.makedirs(os.path.join(args.root_dir, 'sparse'), exist_ok=True)
        os.system(f'''
            CUDA_VISIBLE_DEVICES={args.cuda_device} colmap mapper \
                --database_path={args.root_dir}/database.db \
                --image_path={args.root_dir}/{args.images_resized} \
                --output_path={args.root_dir}/sparse ''')

    undist_dir = os.path.join(args.root_dir, args.undistorted_output)
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)
        os.system(f'''
            CUDA_VISIBLE_DEVICES={args.cuda_device} colmap image_undistorter \
                --input_path={args.root_dir}/sparse/0 \
                --image_path={args.root_dir}/{args.images_resized} \
                --output_path={args.root_dir} \
                --output_type=COLMAP''')

def generate_depth(args):
    disp_dir = os.path.join(args.root_dir, 'disps')
    if not os.path.exists(disp_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/depth')
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        #os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
        # os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run.py --Final --data_dir {args.root_dir}/images --output_dir {args.root_dir}/disps --depthNet 0')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run_monodepth.py -i {args.root_dir}/images -o {args.root_dir}/disps -t dpt_large')
        os.chdir(f'{str(cur_dir)}')

def generate_flow(args):
    flow_fw_dir = os.path.join(args.root_dir, 'flow_fw')
    flow_bw_dir = os.path.join(args.root_dir, 'flow_bw')
    if not os.path.exists(flow_fw_dir) or not os.path.exists(flow_bw_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/flow')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python demo.py --model models/raft-things.pth --path {args.root_dir}')
        os.chdir(f'{str(cur_dir)}')

if __name__ == '__main__':
    args = parse_args()

    resize_frames(args)
    generate_masks(args)
    run_colmap(args)
    generate_depth(args)
    generate_flow(args)
    print('finished!')