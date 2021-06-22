# nsff_pl

Neural Scene Flow Fields using pytorch-lightning. This repo reimplements the [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields) idea, but modifies several operations based on observation of NSFF results and discussions with the authors. For discussion details, please see the [issues](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues?q=is%3Aissue+author%3Akwea123) of the original repo. The code is based on my previous [implementation](https://github.com/kwea123/nerf_pl).

The main modifications are the followings:

1.  **Remove the blending weight in static NeRF. I adopt the addition strategy in [NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw).**
2.  **Remove disocclusion head. I use warped dynamic weights as an indicator of whether occlusion occurs.** At the beginning of training, this indicator acts reliably as shown below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/122887190-9f2e9380-d37b-11eb-8bbf-f11d22707a16.png", width="80%">
  <br>
  <sup>Top: Reference image. Center: Warped images, artifacts appear at boundaries. Bottom: Estimated disocclusion.</sup>
</p>
As training goes, the disocclusion tends to get close to 1, i.e. occlusion does not exist even in warping. In my opinion, this means the empty space learns to "move a little" to avoid the space occupied by dynamic objects.

3.  **Compose static dynamic also in image warping.**

Implementation details are in [models/rendering.py](models/rendering.py).

The implementation is verified on several sequences, and produces visually plausible results. Qualitatively, these modifications produces better result on the `kid-running` scene compared to the original repo.

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

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/122886630-1879b680-d37b-11eb-8265-7a418d44d24a.gif", width="40%">
</p>

### Time interpolation

<p align="center">
  <img src="assets/kid_fv8.gif", width="40%">
</p>

The color of our method is more vivid and closer to the GT images both qualitatively and quantitatively (not because of gif compression). Also, the background is more stable and cleaner.

<!-- ### Bonus - Depth

Our method also produces smoother depths, although it might not have direct impact on image quality.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/121126332-f5231780-c862-11eb-89a6-98558e479c69.png", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/121126404-0ff58c00-c863-11eb-9b31-72824b944ed1.png", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/121126294-e9375580-c862-11eb-8a36-e560b4318621.png", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/121126457-2865a680-c863-11eb-8fbd-aad471efbe3d.png", width="40%">
  <br>
  <sup>Top left: static depth from this repo. Top right: full depth from this repo. <br> Bottom left: static depth from the original repo. Bottom right: full depth from the original repo.
  </sup>
</p> -->

### More results


# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti)

## Software

* Clone this repo by `git clone --recursive https://github.com/kwea123/nsff_pl`
* Python>=3.7 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n nsff_pl python=3.7` to create a conda environment and activate it by `conda activate nsff_pl`)
* Install core requirements by `pip install -r requirements.txt`
* Install `cupy` via `pip install cupy-cudaxxx` by replacing `xxx` with your cuda version.

# :key: Training

<details>
  <summary>Steps</summary>

## Data preparation

Create a root directory (e.g. `foobar`), create a folder named `frames` and prepare your images (it is recommended to have at least 30 images) under it, so the structure looks like:

```bash
└── foobar
    └── frames
        ├── 00000.png
        ...
        └── 00029.png
```

The image names can be arbitrary, but the lexical order should be the same as time order! E.g. you can name the images as `a.png`, `c.png`, `dd.png` but the time order must be `a -> c -> dd`.

### Motion Mask

In order to correctly reconstruct the camera poses, we must first filter out the dynamic areas so that feature points in these areas are not matched during estimation.

I use maskrcnn from [detectron2](https://github.com/facebookresearch/detectron2). Only semantic masks are used, as I find flow-based masks too noisy.

Install detectron2 by `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html`.

Modify the `DYNAMIC_CATEGORIES` variable in `third_party/predict_mask.py` to the dynamic classes in your data (only COCO classes are supported).

Next, NSFF requires depth and optical flows. We'll use some SOTA methods to perform the prediction.
  
### Depth
  
The instructions and code are borrowed from [DPT](https://github.com/intel-isl/DPT).

Download the model weights from [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and put it in `third_party/depth/weights/`.

### Optical Flow
The instructions and code are borrowed from [RAFT](https://github.com/princeton-vl/RAFT).

Download `raft-things.pth` from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and put it in `third_party/flow/models/`.

### Prediction
  
Thanks to [owang](https://github.com/owang), after preparing the images and the model weights, we can automate the whole process by a single command `python preprocess.py --root_dir <path/to/foobar>`.

Finally, your root directory will have all of this:

```bash
└── foobar
    ├── frames (original images, not used, you can delete)
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── images_resized (resized images, not used, you can delete)
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── images (the images to use in training)
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── masks (not used but do not delete)
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
  
## Train!

Run the following command (modify the parameters according to `opt.py`):
```j
python train.py \
  --dataset_name monocular --root_dir $ROOT_DIR \
  --img_wh 512 288 --start_end 0 30 \
  --N_samples 128 --N_importance 0 --encode_t --use_viewdir \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name exp
```
  
</details>

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

More specifically, the `split` argument specifies which novel view to generate:
*  `test`: test on training pose and times
*  `test_spiral`: spiral path over the whole sequence, with time gradually advances (integer time for now)
*  `test_spiralX`: fix the time to `X` and generate spiral path around training view `X`.
*  `test_fixviewX_interpY`: fix the view to training pose `X` and interpolate the time from start to end, adding `Y` frames between each integer timestamps.

# :warning: Other differences with the original paper

1.  I add entropy loss as suggested [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues/18#issuecomment-851038816). This allows the person to be "thin" and produces less artifact when the camera is far from the original pose.
2.  I explicitly zero the flow at far regions (where `z>0.95`).
3.  I add a cross entropy loss to encourage static and dynamic weights to peak at different locations (i.e. one sample points is either static or dynamic).

# TODO
- [x] Add COLMAP reconstruction tutorial (mask out dynamic region).
- [x] Remove NSFF dependency for data preparation. More precisely, the original code needs quite a lot modifications to work on own data, and the depth/flow are calculated on resized images, which might reduce their accuracy.
- [x] Add spiral path for testing.
- [ ] Add mask hard mining at the beginning of training. Or prioritized experience replay.

# Acknowledgment

Thank to the authors of the NSFF paper, [owang](https://github.com/owang) [zhengqili](https://github.com/zhengqili) [sniklaus](https://github.com/sniklaus), for fruitful discussions and supports!
