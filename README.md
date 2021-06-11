# nsff_pl
Neural Scene Flow Fields using pytorch-lightning. This repo reimplements the [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields) idea, but modifies several operations based on observation of NSFF results and discussions with the authors. For discussion details, please see the [issues](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues?q=is%3Aissue+author%3Akwea123) of the original repo. The code is based on my previous [implementation](https://github.com/kwea123/nerf_pl).

The main modifications are the followings:

1.  **Remove the blending weight in static NeRF. I adopt the addition strategy in [NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw).**
2.  **Compose static dynamic also in image warping.**

Implementation details are in [models/rendering.py](models/rendering.py).

These modifications empirically produces better result on the `kid-running` scene, as shown below:

**IMPORTANT**: The code for `kid-running` scene is moved to [nsff_orig](https://github.com/kwea123/nsff_pl/tree/nsff_orig) branch (the images are still shown here just to showcase)! The `master` branch will be updated for custom data usage.

### Full reconstruction

<p align="center">
  <img src="assets/recon.gif", width="99%">
  <br>
  <sup>Left: GT. Center: this repo (PSNR=35.02). Right: pretrained model of the original repo(PSNR=30.45).</sup>
</p>

### Background reconstruction

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/121126826-c194bd00-c863-11eb-9e36-a4790455df2f.gif", width="80%">
  <br>
  <sup>Left: this repo. Right: pretrained model of the original repo (by setting raw_blend_w to 0).</sup>
</p>

### Fix-view-change-time (view 8, times from 0 to 16)

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/121122112-d3726200-c85b-11eb-8aaf-b4757a035280.gif", width="80%">
  <br>
  <sup>Left: this repo. Right: pretrained model of the original repo.</sup>
</p>

### Fix-time-change-view (time 8, views from 0 to 16)

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/121125017-cd32b480-c860-11eb-9cbf-e96674967963.gif", width="80%">
  <br>
  <sup>Left: this repo. Right: pretrained model of the original repo.</sup>
</p>

### Novel view synthesis (spiral)

The color of our method is more vivid and closer to the GT images both qualitatively and quantitatively (not because of gif compression). Also, the background is more stable and cleaner.

### Bonus - Depth

Our method also produces smoother depths, although it might not have direct impact on image quality.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/121126332-f5231780-c862-11eb-89a6-98558e479c69.png", width="48%">
  <img src="https://user-images.githubusercontent.com/11364490/121126404-0ff58c00-c863-11eb-9b31-72824b944ed1.png", width="48%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/121126294-e9375580-c862-11eb-8a36-e560b4318621.png", width="48%">
  <img src="https://user-images.githubusercontent.com/11364490/121126457-2865a680-c863-11eb-8fbd-aad471efbe3d.png", width="48%">
  <br>
  <sup>Top left: static depth from this repo. Top right: full depth from this repo. <br> Bottom left: static depth from the original repo. Bottom right: full depth from the original repo.
  </sup>
</p>

⚠️ However, more experiments on other scenes are needed to finally prove that these modifications produce overall better quality.


# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti)

## Software

* Clone this repo by `git clone --recursive https://github.com/kwea123/nsff_pl`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n nsff_pl python=3.6` to create a conda environment and activate it by `conda activate nsff_pl`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`

# :key: Training

## 0. Data preparation

~~The data preparation follows the original repo. Therefore, please follow [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields#video-preprocessing) to prepare the data (resized images, monodepth and flow) for training.~~ If your data format follows the original repo or use the `kid-running` sequence, please use [nsff_orig](https://github.com/kwea123/nsff_pl/tree/nsff_orig) branch.

Otherwise, create a root directory (e.g. `foobar`), create a folder named `images` and prepare your images (it is recommended to have at least 30 images) under it, so the structure looks like:

```bash
└── foobar
    └── images
        ├── 00000.png
        ...
        └── 00029.png
```

Save the root directory as an environment variable to simplify the code in the following processes:
```bash
export ROOT_DIR=/path/to/foobar/
```

The image names can be arbitrary, but the lexical order should be the same as time order! E.g. you can name the images as `a.png`, `c.png`, `dd.png` but the time order must be `a -> c -> dd`. Only one constraint: **do not** put the string "images" in the image name! e.g. `images001.png` is ❎!

## 1. Motion mask prediction and COLMAP pose reconstruction

### Motion mask prediction

In order to correctly reconstruct the camera poses, we must first filter out the dynamic areas so that feature points in these areas are not matched during estimation.

I use maskrcnn from [detectron2](https://github.com/facebookresearch/detectron2). Only semantic masks are used, as I find flow-based masks too noisy.

Install detectron2 by `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html`.

Modify the `DYNAMIC_CATEGORIES` variable in `third_party/predict_mask.py` to the dynamic classes in your data (only COCO classes are supported). Run `python third_party/predict_mask.py --root_dir $ROOT_DIR`. After that, your root directory will contain motion masks (0=dynamic and 1=static):

```bash
└── foobar
    ├── images
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    └── masks
        ├── 00000.png.png
        ...
        └── 00029.png.png
```

The masks need not be perfect, they can mask static regions, but most of the dynamic regions MUST lie inside the mask. If not, try lowering the `DETECTION_THR` or changing the `DYNAMIC_CATEGORIES`.

### COLMAP pose reconstruction

Please first install COLMAP following the [official tutorial](https://colmap.github.io/install.html).

Here I only briefly explain how to reconstruct the poses using GUI. For command line usage, please search by yourself.

1.  Run `colmap gui`.
2.  Select the tab `Reconstruction -> Automatic reconstruction`.
3.  Select "Workspace folder" as `foobar`, "Image folder" as `foobar/images`, "Mask folder" as `foobar/masks`.
4.  Select "Data type" as "Video frames".
5.  Check "Shared intrinsics" and uncheck "Dense model".
6.  Press "Run".

After reconstruction, you should see reconstructed camera poses as red quadrangular pyramids, and some reconstructed point clouds. Please roughly judge if the poses are correct (e.g. if your camera moves forward, but COLMAP reconstructs horizontal movements, then this is incorrect), if not, consider retake the photos.

Now your root directory should look like:

```bash
└── foobar
    ├── images
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── masks
    │   ├── 00000.png.png
    │   ...
    │   └── 00029.png.png
    ├── database.db
    └── sparse
        └── 0
            ├── cameras.bin
            ├── images.bin
            ├── points3D.bin
            └── project.ini
```

## 2. Monodepth and optical flow prediction

### Monodepth
The instructions and code are borrowed from [BoostingMonocularDepth](https://github.com/compphoto/BoostingMonocularDepth).

1.  Download the mergenet model weights from [here](https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth) and put it in `third_party/depth/pix2pix/checkpoints/mergemodel/`.

2.  Download the model weights from [MiDas-v2](https://github.com/intel-isl/MiDaS/tree/v2) and put it in `third_party/depth/midas/`.

3.  From `thrid_party/depth`, run `python run.py --Final --data_dir $ROOT_DIR/images --output_dir $ROOT_DIR/disps --depthNet 0`

It will create 16bit depth images under `$ROOT_DIR/disps`. This monodepth method is more accurate than most of the SOTA method, so it takes a few seconds to process each image.

### RAFT
The instructions and code are borrowed from [RAFT](https://github.com/princeton-vl/RAFT).

1.  Download `raft-things.pth` from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and put it in `third_party/flow/models/`.

2.  From `third_party/flow/`, run `python demo.py --model models/raft-things.pth --path $ROOT_DIR`.

Finally, your root directory will have all of this:

```bash
└── foobar
    ├── images
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── masks
    │   ├── 00000.png.png
    │   ...
    │   └── 00029.png.png
    ├── database.db
    ├── sparse
    │   └── 0
    │       ├── cameras.bin
    │       ├── images.bin
    │       ├── points3D.bin
    │       └── project.ini
    ├── disps
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── flow_fw
    │   ├── 00000.flo
    │   ...
    │   └── 00028.flo
    └── flow_bw
        ├── 00001.flo
        ...
        └── 00029.flo
```

Now you can start training!

## 3. Train!

Run the following command (modify the parameters according to `opt.py`):
```j
python train.py \
  --dataset_name monocular --root_dir $ROOT_DIR \
  --img_wh 512 288 --start_end 0 30 --batch_from_same_image \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name exp
```

## Comparison with other repos

|           | training GPU memory in GB (batchsize=512) | speed (1 step) | training time/final PSNR on kid-running |
| :---:     |  :---:     | :---:   | :---: |
| [Original](https://github.com/zhengqili/Neural-Scene-Flow-Fields)  | 7.6 | **0.2s** | 96 GPUh / 30.45 |
| This repo | **5.9** | **0.2s** | **12 GPUh / 35.02**

The speed is measured on 1 RTX2080Ti.

# :mag_right: Testing

See [test.ipynb](https://nbviewer.jupyter.org/github/kwea123/nsff_pl/blob/master/test.ipynb) for scene reconstruction, scene decomposition, fix-time-change-view, ..., etc. You can get almost everything out of this notebook.
I will add more instructions inside in the future.

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```j
python eval.py \
  --dataset_name monocular --root_dir $ROOT_DIR \
  --N_samples 128 --N_importance 0 --img_wh 512 288 --start_end 0 30 \
  --encode_t --output_transient \
  --split test --video_format gif --fps 5 \
  --ckpt_path kid.ckpt --scene_name kid_reconstruction
```

# :warning: Other differences with the original paper

1.  I add entropy loss as suggested [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues/18#issuecomment-851038816). This allows the person to be "thin" and produces less artifact when the camera is far from the original pose.
2.  I explicitly zero the flows of far regions to avoid the flow being trapped in local minima (reason explained [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues/19#issuecomment-855958276)).

# TODO
- [x] Add COLMAP reconstruction tutorial (mask out dynamic region).
- [x] Remove NSFF dependency for data preparation. More precisely, the original code needs quite a lot modifications to work on own data, and the depth/flow are calculated on resized images, which might reduce their accuracy.
- [x] Add spiral path for testing.
- [ ] Add mask hard mining at the beginning of training.
- [ ] Exploit motion mask prior like https://free-view-video.github.io/
