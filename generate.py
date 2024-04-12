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


BROWN = (147, 107, 76)
GOLD  = (212, 159, 65)
BEIGE = (249, 246, 227)
WHITE = (255, 255, 255)


class MarkerType(Enum):
    BROWN = 0
    GOLD  = 1
    BEIGE = 2
    WHITE = 3


def make_circle(size):
    r = size / 2
    mat = np.zeros((size, size), dtype=np.uint8)
    dist_from_center = np.sqrt((np.arange(size) - r) ** 2 + (np.arange(size)[:, None] - r) ** 2)
    mat[dist_from_center <= r] = 255
    return mat


def make_marker(marker, size):
    mat = np.zeros((size, size, 3), dtype=np.uint8)
    if marker == MarkerType.BROWN:
        mat[:, :] = WHITE
        mat[make_circle(size) != 0] = BROWN
    elif marker == MarkerType.GOLD:
        mat[make_circle(size) != 0] = GOLD
    elif marker == MarkerType.BEIGE:
        mat[make_circle(size) != 0] = BEIGE
    elif marker == MarkerType.WHITE:
        mat[:, :] = WHITE
    else:
        raise ValueError(f'Unknown marker type: {marker}')
    return mat


texture_transform = A.Compose([
    # A.RandomBrightnessContrast(p=1),
    # A.RandomGamma(p=1),
    # A.RGBShift(p=1),
])


image_transform = A.Compose([
    A.Resize(640, 640),
])


shape_transform = A.Compose([
    A.Affine(p=1, scale=(0.03, 0.1), rotate=(0, 90), keep_ratio=True, interpolation=Image.NONE),
    A.Perspective(p=1, interpolation=Image.NONE),
    A.Affine(
        p=1,
        translate_percent=(-0.4, 0.4),
        interpolation=Image.NONE,
    ),
])


def add_circles(image, _):
    mat = np.array(image)
    mat = image_transform(image=mat)['image']
    
    targets = []
    for _ in range(random.randint(0, 6)):
        marker_class = random.randint(0, 3)
        marker_type = MarkerType(marker_class)

        marker = make_marker(marker_type, 640)
        marker_mask = (marker != 0).any(axis=2)
        
        marker = texture_transform(image=marker)['image']
        marker[~marker_mask] = 0

        marker = shape_transform(image=marker)['image']

        marker_mask = (marker != 0).any(axis=2)
        mat[marker_mask] = marker[marker_mask]

        bbox = np.where(marker_mask)

        if len(bbox[0]) == 0 or len(bbox[1]) == 0:
            continue

        y1, x1 = bbox[0].min(), bbox[1].min()
        y2, x2 = bbox[0].max(), bbox[1].max()
        w, h = x2 - x1, y2 - y1

        if w * h < 100:
            continue

        targets.append({
            'category_id': marker_class,
            'bbox': list(map(float, [x1, y1, w, h])),
        })
    
    image = Image.fromarray(mat)
    return image, targets


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
                    f.write(f'{t["category_id"]} {x} {y} {w} {h}\n')
        with ProgressBar():
            db.from_sequence(range(len(data)), ).map(save_image).compute(progress=True)
    with open(os.path.join(path, 'data.yaml'), 'w') as f:
        yaml.dump({
            'names': ['brown', 'gold', 'beige', 'white'],
            'nc': 4,
            **{split: os.path.join(path, split) for split in splits},
        }, f)


if __name__ == '__main__':
    dataset = CocoDetection(
        root='coco/val2017',
        annFile='coco/annotations/instances_val2017.json',
        transforms=add_circles,
    )

    train, val, _ = random_split(dataset, [500, 100, 4400], generator=torch.Generator().manual_seed(42))

    save_yolo_dataset('yolo_circles', train=train, val=val)
