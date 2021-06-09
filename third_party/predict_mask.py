import argparse
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

DETECTION_THR = 0.05
DYNAMIC_CATEGORIES = [
    'person', 'book', 'remote', 'parking meter',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')

    args = parser.parse_args()

    print("Motion classes are :", DYNAMIC_CATEGORIES)
    print("Please modify the DYNAMIC_CATEGORIES variables to your need!")
    os.makedirs(os.path.join(args.root_dir, 'masks'), exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THR
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    print("Predicting motion masks ...")
    for img_p in tqdm(sorted(glob.glob(os.path.join(args.root_dir, 'images/*')))):
        img = cv2.imread(img_p)
        outputs = predictor(img)
        motion_mask = 255*np.ones_like(img)

        for i in range(len(outputs['instances'])):
            if COCO_INSTANCE_CATEGORY_NAMES[outputs['instances'].pred_classes[i]] \
                in DYNAMIC_CATEGORIES:
                motion_mask[outputs['instances'].pred_masks[i, :, :].cpu().numpy()] = 0

        # enlarge the mask a little to account for inaccurate boundaries
        motion_mask = cv2.erode(motion_mask, np.ones((15, 15), np.uint8), iterations=1)
        cv2.imwrite(img_p.replace('images', 'masks'), motion_mask)