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

As training goes, the disocclusion tends to get close to 1 almost everywhere, i.e. occlusion does not exist even in warping. In my opinion, this means the empty space learns to "move a little" to avoid the space occupied by dynamic objects (although the network has *never* been trained to do so).

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

### Novel view synthesis (view 8, spiral)

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/122886630-1879b680-d37b-11eb-8265-7a418d44d24a.gif", width="40%">
</p>

### Time interpolation (view 8, add 10 frames between each integer time from time 0 to 29)

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/122915609-e0806c80-d396-11eb-934d-65f9d107b5ce.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/123046460-170dc400-d437-11eb-9ac9-29086e438062.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/123046853-8f748500-d437-11eb-9e46-7b634bb15c69.png", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/123046793-7ff53c00-d437-11eb-9f9a-e3c46ce207f4.png", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/123046862-91d6df00-d437-11eb-8fa5-aaea58c1a643.png", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/123046797-81beff80-d437-11eb-9e68-6cfb95310d6d.png", width="40%">
  <br>
  <sup>Left: this repo. Right: pretrained model of the original repo. The 2nd and 3rd rows are 0th frame and 29th frame to show the difference of the background.</sup>
</p>

The color of our method is more vivid and closer to the GT images both qualitatively and quantitatively (not because of gif compression). Also, even **without** any kind of supervision (either direct or self supervision), the network learns to separate the foreground and the background more cleanly than the original implementation, which is unexpected! Bad fg/bg separation not only means the background actually *changes* each frame, but also the color information is not leverage across time, so the reconstruction quality degrades, as can be shown in the original NSFF result towards the end.

### Bonus - Depth

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
</p>

### More results

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/122894224-f6cffd80-d381-11eb-96b8-25a6d9136dbf.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/122894158-ea4ba500-d381-11eb-9dd9-1292b892bba9.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/122915244-82ec2000-d396-11eb-98d1-fc58bdd6fbf7.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/122915260-87b0d400-d396-11eb-8159-5c6ac9caabfd.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/122893961-bf615100-d381-11eb-9171-ec19c3cd3832.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/122894027-d011c700-d381-11eb-9d7e-213955702634.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/122928853-3825d480-d3a5-11eb-94cc-5ba07127b05f.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/122928862-39ef9800-d3a5-11eb-8f06-9802e20f0414.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/122926467-d7959800-d3a2-11eb-9b6a-253d1509afb2.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/122926576-f7c55700-d3a2-11eb-90a6-70340ec6d0ce.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/123185944-052d2f00-d4d2-11eb-8093-13d1cabc2f2d.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/123185950-06f6f280-d4d2-11eb-8d12-7c1f0cdbf39b.gif", width="40%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/123023518-773e3f00-d412-11eb-9dc7-eb91329c414b.gif", width="40%">
  <img src="https://user-images.githubusercontent.com/11364490/123023585-989f2b00-d412-11eb-94d6-685b0cedf417.gif", width="40%">
  <br>
  </sup>
</p>

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

I also implemented a hard sampling strategy to improve the quality of the hard regions. Add `--hard_sampling` to enable it.

Specifically, I compute the SSIM between the prediction and the GT at the end of each epoch, and use **1-SSIM** as the sampling probability for the next epoch. This allows rays with larger errors to be sampled more frequently, and thus improve the result. The choice of SSIM is that it reflects more visual quality, and is less sensible to noise or small pixel displacement like PSNR.
  
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
3.  I add a cross entropy loss with thickness to encourage static and dynamic weights to peak at different locations (i.e. one sample points is either static or dynamic). The thickness specifies how many intervals should the peaks be separated. Empirically I found 15 to be a good value. Explanation about this loss is, for example for the "kid playing bubble scene", the kid is rotating around himself with the rotation axis almost fixed, so if no prior is added, the network learns the central part around the body as **static** (for example you can imagine a flag rotating around its pole, only the flag moves but the pole doesn't). This doesn't cause problem for reconstruction, but when it comes to novel view, the wrongly estimated static part causes artifacts. In order to make the network learn that the whole body is moving, I add this cross entropy loss to force the static peak to be at least `thickness//2` far from the dynamic peak.
4.  In pose reconstruction, the original authors use **the entire image** to reconstruct the poses, without masking out the dynamic region. In my opinion this strategy might lead to totally wrong pose estimation in some cases, so I opt to reconstruct the poses with dynamic region masked out. In order to set the near plane correctly, I use COLMAP combined with monodepth to get the minimum depth.

# TODO
- [x] Add COLMAP reconstruction tutorial (mask out dynamic region).
- [x] Remove NSFF dependency for data preparation. More precisely, the original code needs quite a lot modifications to work on own data, and the depth/flow are calculated on resized images, which might reduce their accuracy.
- [x] Add spiral path for testing.
- [x] Add mask hard mining at the beginning of training. Or prioritized experience replay.

# Acknowledgment

Thank to the authors of the NSFF paper, [owang](https://github.com/owang) [zhengqili](https://github.com/zhengqili) [sniklaus](https://github.com/sniklaus), for fruitful discussions and supports!
