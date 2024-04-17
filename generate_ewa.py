import numpy as np
import torch
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import albumentations as A
import os
import shutil
import dask.bag as db
from dask.diagnostics import ProgressBar
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
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image_path = os.path.join(path, split, 'images', f'{i}.jpg')
            image.save(image_path)

            with open(os.path.join(path, split, 'labels', f'{i}.txt'), 'w') as f:
                for t in target:
                    
                    # spłaszczona lista keypoints
                    bbox = []
                    for tuple in target['keypoints']:
                        bbox.extend(tuple)

                    x_center = (bbox[0] + bbox[2]) / (2 * image.width)
                    y_center = (bbox[1] + bbox[3]) / (2 * image.height)
                    width = (bbox[2] - bbox[0]) / image.width
                    height = (bbox[3] - bbox[1]) / image.height
                    f.write(f'0 {x_center} {y_center} {width} {height} {bbox[0]} {bbox[1]} 2 {bbox[2]} {bbox[3]} 2\n')

        # Zapisywanie danych partiami
        batch_size = 1000
        num_batches = len(data) // batch_size + 1
        for batch_idx in tqdm(range(num_batches), desc=f'Saving {split} set'):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            for idx in range(len(batch_data)):
                save_image(start_idx + idx)

    with open(os.path.join(path, 'data.yaml'), 'w') as f:
        yaml.dump({
            'names': ['needle'],
            'nc': 1,
            **{split: os.path.join(path, split) for split in splits},
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
        root='needle-student-keypoint-2/train',
        annFile='needle-student-keypoint-2/train/_annotations.coco.json',
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



