import cv2
from pathlib import Path


if __name__ == '__main__':
    root = Path('experiments')
    image_files = list(root.glob('*.png'))

    for image_file in image_files:
        image = cv2.imread(str(image_file))

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f'x: {x}, y: {y}')
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('image', image)
                with open(image_file.with_suffix('.txt'), 'w') as f:
                    f.write(f'{x} {y}\n')

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', image)
        cv2.waitKey(0)
