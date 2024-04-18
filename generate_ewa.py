import numpy as np
import torch
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import albumentations as A
import os
import shutil
import yaml
from tqdm import tqdm
from PIL import Image
import cv2



def transform(x):
    return np.array(x)

def target_transform(x):
    kpt = x[0]['keypoints']
    return [(x, y) for x, y, v in zip(kpt[0::3], kpt[1::3], kpt[2::3])]

def augment_data(image, keypoints):
    # wywoływane dla jednego zdjęcia, a na zewnątrz będzie pętla for

    transform = A.Compose([
        A.Crop(0, 500, image.shape[1], 1600),
        A.Resize(640, 640),
        A.Rotate(p=1),
        A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1)),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transformed = transform(
        image=image,
        keypoints=keypoints,
    )

    # zwraca zdjęcie i keypoints, ta funkcja nie musi zwracać bbox
    return transformed['image'], {'keypoints': transformed['keypoints']}


def transform_test_data(image, keypoints):
    # wywoływane dla jednego zdjęcia, a na zewnątrz będzie pętla for

    transform = A.Compose([
        A.Crop(0, 500, image.shape[1], 1600),
        A.Resize(640, 640),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transformed = transform(
        image=image,
        keypoints=keypoints,
    )

    # zwraca zdjęcie i keypoints, ta funkcja nie musi zwracać bbox
    return transformed['image'], {'keypoints': transformed['keypoints']}



def save_yolo_dataset(path, **splits):
    shutil.rmtree(path, ignore_errors=True)
    for split, data in splits.items():
        os.makedirs(os.path.join(path, split, 'images'))
        os.makedirs(os.path.join(path, split, 'labels'))

        def save_image(i):
            image, target = data[i]
            # konwersja obrazu do PIL
            image = Image.fromarray(image)

            image_path = os.path.join(path, split, 'images', f'{i}.jpg')

            p0x, p0y = target['keypoints'][0]
            p1x, p1y = target['keypoints'][1]

            p0x /= image.width
            p0y /= image.height
            p1x /= image.width
            p1y /= image.height

            if p0x < 0 or p0y < 0 or p1x < 0 or p1y < 0 or p0x > 1 or p0y > 1 or p1x > 1 or p1y > 1:
                return

            print('Saving', image_path)
            image.save(image_path)

            x_center = (p0x + p1x) / 2
            y_center = (p0y + p1y) / 2

            width = abs(p0x - p1x) + 0.01
            height = abs(p0y - p1y) + 0.01

            with open(os.path.join(path, split, 'labels', f'{i}.txt'), 'w') as f:
                f.write(f'0 {x_center} {y_center} {width} {height} {p0x} {p0y} {p1x} {p1y}\n')
                # f.write(f'0 {x_center} {y_center} {width} {height}\n')

        for i in tqdm(range(len(data)), desc=f'Saving {split} set'):
                save_image(i)

    with open(os.path.join(path, 'data.yaml'), 'w') as f:
        yaml.dump({
            'names': ['needle'],
            'nc': 1,
            'kpt_shape': [2, 2],
            'flip_idx': [0, 1],
            **{split: os.path.join(split) for split in splits},
        }, f)


if __name__ == '__main__':

    # odczytanie danych z formatu COCO
    train_dataset = CocoDetection(
        root='needle-student-keypoint-2/train',
        annFile='needle-student-keypoint-2/train/_annotations.coco.json',
        transform=transform,
        target_transform=target_transform,
    )

    test_dataset = CocoDetection(
        root='needle-student-keypoint-2/valid',
        annFile='needle-student-keypoint-2/valid/_annotations.coco.json',
        transform=transform,
        target_transform=target_transform,
    )

    # augmentacja danych treningowych
    aug_dataset = []
    for image, target in train_dataset:
        for _ in range(10):
            aug_dataset.append(augment_data(image=image, keypoints=target))

    # transformacja danych testowych
    val_dataset = []
    for image, target in test_dataset:
        val_dataset.append(transform_test_data(image=image, keypoints=target))


    # konwersja danych z formatu COCO do formatu YOLO
    save_yolo_dataset('needle_augmented', train=aug_dataset, val=val_dataset)



