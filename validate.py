import argparse
import cv2
import cv2.aruco as aruco
import optuna
import numpy as np
from pathlib import Path
import detect_aruco2


camera_matrix = np.array([[1000.0, 0.0, 500.0],
                        [0.0, 1000.0, 500.0],
                        [0.0, 0.0, 1.0]])

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


def validate(image_files, rvecss, tvecss, needle_xyz):
    results = []

    for i in range(len(image_files)):
        image_file = image_files[i]
        rvecs = rvecss[i]
        tvecs = tvecss[i]

        xy, _ = cv2.projectPoints(needle_xyz, rvecs, tvecs, camera_matrix, dist_coeffs)
        x, y = xy[0][0]

        with open(str(image_file.with_suffix('.txt'))) as f:
            target_x, target_y = map(float, f.read().strip().split())
        error = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        results.append(error)

    loss = np.mean(results)
    return loss


def preprocess(image_files):
    parser = argparse.ArgumentParser(description='Arguments to main')
    parser.add_argument('-a', '--code_side_length', type=int, help='Marker side length', default=10)
    parser.add_argument('-b', '--board_height', type=float, help='Board protrusion height', default=5)
    parser.add_argument('-s', '--spacing', type=float, help='Spacing between aruco markers', default=1)
    parser.add_argument('-d', '--distance', type=float, help='Board distance from the needle', default=0)
    parser.add_argument('-l', '--needle_length', type=float, help='Needle length', default=50)
    args = parser.parse_args()

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    all_obj_points = detect_aruco2.generate_plus_obj_points(args.code_side_length, args.board_height, args.spacing, args.distance)

    rvecss = []
    tvecss = []

    for image_file in image_files:
        image = cv2.imread(str(image_file))
        
        corners, ids, _ = detector.detectMarkers(image)

        if ids is None:
            assert False, 'No markers found'

        obj_points = []
        image_points = []
        for id in range(5):
            if id not in ids:
                continue
            i = np.where(ids == id)[0][0]
            obj_points.extend(all_obj_points[id])
            image_points.extend(corners[i][0])
        obj_points = np.array(obj_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        _, rvecs, tvecs = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)

        rvecss.append(rvecs)
        tvecss.append(tvecs)

    return rvecss, tvecss


if __name__ == '__main__':

    root = Path('experiments')
    image_files = sorted(list(root.glob('*.png')))

    rvecss, tvecss = preprocess(image_files)

    def objective(trial: optuna.Trial):
        needle_x = trial.suggest_float('needle_x', -500, 0)
        needle_y = trial.suggest_float('needle_y', -100, 100)
        needle_z = trial.suggest_float('needle_z', -100, 100)
        needle_xyz = np.array([[needle_x, needle_y, needle_z]], dtype=np.float32)
        loss = validate(image_files, rvecss, tvecss, needle_xyz)
        return loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5000)
    print(study.best_params)
    print(study.best_value)
