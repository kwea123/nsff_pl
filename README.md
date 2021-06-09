# nsff_pl
Neural Scene Flow Fields using pytorch-lightning. This repo reimplements the [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields) idea, but modifies several operations based on observation of NSFF results and discussions with the authors. For discussion details, please see the [issues](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues?q=is%3Aissue+author%3Akwea123) of the original repo. The code is based on my previous [implementation](https://github.com/kwea123/nerf_pl).

The main modifications are the followings:

1.  **Remove the blending weight in static NeRF. I adopt the addition strategy in [NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw).**
2.  **Compose static dynamic also in image warping.**

Implementation details are in [models/rendering.py](models/rendering.py).

These modifications empirically produces better result on the `kid-running` scene, as shown below:

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

## Data preparation

~~The data preparation follows the original repo. Therefore, please follow [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields#video-preprocessing) to prepare the data (resized images, monodepth and flow) for training.~~ If your data format follows the original repo or use the `kid-running` sequence, please use [nsff_orig](https://github.com/kwea123/nsff_pl/tree/nsff_orig) branch.

## COLMAP pose reconstruction

TODO

## Monodepth and optical flow prediction

TODO

## Train!

Run the following command:
```j
python train.py \
  --dataset_name monocular --root_dir $ROOT_DIR \
  --img_wh 512 288 --start_end 0 30 --batch_from_same_image \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name exp 
```
where `$ROOT_DIR` is where the data is located.

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nsff_pl/releases).

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
  --dataset_name monocular --root_dir /home/ubuntu/data/nerf_example_data/my/kid-running/dense \
  --N_samples 128 --N_importance 0 --img_wh 512 288 --start_end 0 30 \
  --encode_t --output_transient \
  --split test --video_format gif --fps 5 \
  --ckpt_path kid.ckpt --scene_name kid_reconstruction
```

# :warning: Other differences with the original paper

1.  I add entropy loss as suggested [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues/18#issuecomment-851038816). This allows the person to be "thin" and produces less artifact when the camera is far from the original pose.
2.  I explicitly zero the flows of far regions to avoid the flow being trapped in local minima (reason explained [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues/19#issuecomment-855958276)).

# TODO
- [ ] Add COLMAP reconstruction tutorial (mask out dynamic region).
- [ ] Remove NSFF dependency for data preparation. More precisely, the original code needs quite a lot modifications to work on own data, and the depth/flow are calculated on resized images, which might reduce their accuracy.
- [ ] Exploit motion mask prior like https://free-view-video.github.io/
