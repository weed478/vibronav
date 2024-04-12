import numpy as np
from PIL import Image
import albumentations as A
import random
from enum import Enum
from torchvision.datasets import CocoDetection
import torch
from torch.utils.data import random_split
import yaml
import os
import shutil
from tqdm import tqdm
import dask.bag as db
from dask.diagnostics import ProgressBar


a_train_transform = A.Compose([
    A.Crop(0, 500, img.shape[1], 1600),
    A.Resize(640, 640),
    A.Rotate(p=1),
    A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1)),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


a_test_transform = A.Compose([
    A.Crop(0, 500, img.shape[1], 1600),
    A.Resize(640, 640),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def train_transforms(img, ann):
    img = np.array(img)
    kpts = ann[0]['keypoints']
    kpts = [(x, y) for x, y, v in zip(kpts[0::3], kpts[1::3], kpts[2::3])]
    transformed = a_train_transform(image=img, keypoints=kpts[0]['keypoints'])
    return transformed['image'], {'keypoints': transformed['keypoints'], 'bbox': ...}


def save_yolo_dataset(path, **splits):
    shutil.rmtree(path, ignore_errors=True)
    for split, data in splits.items():
        print(f'Saving {split} set')
        os.makedirs(os.path.join(path, split, 'images'))
        os.makedirs(os.path.join(path, split, 'labels'))
        def save_image(i):
            image, target = data[i]
            image_path = os.path.join(path, split, 'images', f'{i}.jpg')
            image.save(image_path)
            with open(os.path.join(path, split, 'labels', f'{i}.txt'), 'w') as f:
                for t in target:
                    x1, y1, w, h = t['bbox']
                    x, y = x1 + w / 2, y1 + h / 2
                    x, y, w, h = x / image.width, y / image.height, w / image.width, h / image.height
                    f.write(f'0 {x} {y} {w} {h} {kpt0x} {kpt0y} 2 {kpt1x} {kpt1y} 2\n')
        with ProgressBar():
            db.from_sequence(range(len(data)), ).map(save_image).compute(progress=True)
    with open(os.path.join(path, 'data.yaml'), 'w') as f:
        yaml.dump({
            'names': ['needle'],
            'nc': 1,
            **{split: os.path.join(path, split) for split in splits},
        }, f)


if __name__ == '__main__':
    train_dataset = CocoDetection(
        root='needle-student-keypoint-2/train',
        annFile='needle-student-keypoint-2/train/_annotations.coco.json',
        transforms=train_transforms,
    )

    aug_dataset = []
    for image, target in train_dataset:
        for _ in range(100):
            aug_dataset.append(augment((image, target)))

    test_dataset = CocoDetection(
        root='needle-student-keypoint-2/valid',
        annFile='needle-student-keypoint-2/valid/_annotations.coco.json',
        transforms=test_transforms,
    )

    save_yolo_dataset('needle_augmented', train=aug_dataset, val=test_dataset)
