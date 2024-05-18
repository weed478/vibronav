import cv2
import cv2.aruco as aruco
import numpy as np
import argparse


def generate_plus_obj_points(code_size, plate_height, spacing):
    points = []

    # positions of top-lef corners of markers in shape "+"
    positions_topleft = [
        (0, 0),                         # central marker
        ((code_size + spacing*2), 0),     # right marker
        (0, (code_size + spacing*2)),     # top marker
        (-(code_size + spacing*2), 0),    # left marker
        (0, -(code_size + spacing*2))     # bottom marker
    ]

    # heights of corners on a plate
    h = plate_height
    heights = [
        (h, h, h, h),
        (h, 0, 0, h),
        (0, 0, h, h),
        (0, h, h, 0),
        (h, h, 0, 0)
    ]

    # generate corner coordinates for each marker
    for i, (x, y) in enumerate(positions_topleft):
        z0, z1, z2, z3 = heights[i]

        points.append([[x, y, z0],
                       [x + code_size, y, z1],
                       [x + code_size, y - code_size, z2],
                       [x, y - code_size, z3]])

    return np.array(points, dtype=np.float32)


def main(a, h, s):
    cap = cv2.VideoCapture(1)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    camera_matrix = np.array([[1000.0, 0.0, 500.0],
                              [0.0, 1000.0, 500.0],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    all_obj_points = generate_plus_obj_points(a, h, s)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            obj_points = []
            image_points = []
            for id in range(len(ids)):
                if id not in ids:
                    continue
                i = np.where(ids == id)[0][0]
                obj_points.extend(all_obj_points[id])
                image_points.extend(corners[i][0])
            obj_points = np.array(obj_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            print('object points: ', obj_points)
            print('image points: ', image_points)

            try:
                _, rvecs, tvecs = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 50)
            except:
                pass

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to main')
    parser.add_argument('a', type=int, help='Code size')
    parser.add_argument('h', type=float, help='Plate height')
    parser.add_argument('s', type=float, help='Spacing between aruco markers')
    args = parser.parse_args()
    main(args.a, args.h, args.s)
