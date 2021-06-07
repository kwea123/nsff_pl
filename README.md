# nsff_pl
Neural Scene Flow Fields using pytorch-lightning. This repo reimplements the [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields) idea, but modifies several operations based on observation of NSFF results and discussions with the authors. For discussion details, please see the [issues](https://github.com/zhengqili/Neural-Scene-Flow-Fields/issues) of the original repo.

The main modifications are the followings:

1.  **Remove the blending weight in static NeRF. I adopt the addition strategy in [NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw).**
2.  **Compose static dynamic also in image warping.**

These modifications empirically produces better result on the `kid-running` scene, as can be shown as below:

⚠️However, more experiments on other scenes are needed to finally prove that these modifications produce overall better quality.

# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti)

## Software

* Clone this repo by `git clone --recursive https://github.com/kwea123/nsff_pl`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n nsff_pl python=3.6` to create a conda environment and activate it by `conda activate nerf_pl`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`

# :key: Training

The data preparation follows the original repo. Therefore, please follow [here](https://github.com/zhengqili/Neural-Scene-Flow-Fields#video-preprocessing) to prepare the data (resized images, monodepth and flow) for training.

After data preparation, run the following command:
```bash
python train.py \
  --dataset_name monocular --root_dir $ROOT_DIR \
  --img_wh 512 288 --start_end 0 30 --batch_from_same_image \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name kid 
```
where `$ROOT_DIR` is where the data is located.

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nsff_pl/releases).

## Comparison with other repos

|           | training GPU memory in GB (batchsize=512) | Speed (1 step) |
| :---:     |  :---:     | :---:   | 
| [Original](https://github.com/zhengqili/Neural-Scene-Flow-Fields)  |  8.5 | 0.177s |
| This repo | 5.9 | 0.2s |

The speed is measured on 1 RTX2080Ti.

# :mag_right: Testing

See [test.ipynb](test.ipynb) for scene reconstruction, scene decomposition, etc.

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py \
   --root_dir $ROOT_DIR \
```
