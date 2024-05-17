import cv2
import cv2.aruco as aruco
import numpy as np
import argparse


# def generate_grid_obj_points(rows, cols, code_size, spacing, heights):
#     points = []
#     for row in range(rows):
#         for col in range(cols):
#             x0 = col * (code_size + spacing)
#             x1 = x0 + code_size
#             y0 = row * (code_size + spacing)
#             y1 = y0 + code_size

#             z0, z1, z2, z3 = heights[row * cols + col]

#             points.append([x0, y0, z0])
#             points.append([x1, y0, z1])
#             points.append([x1, y1, z2])
#             points.append([x0, y1, z3])
#     return np.array(points, dtype=np.float32)


def generate_ring_obj_points(radius, num_codes, code_size, heights):
    circumference = 2 * np.pi * radius
    code_angle = code_size / circumference * 2 * np.pi
    points = []
    for i in range(num_codes):
        angle0 = 2 * np.pi * i / num_codes
        angle1 = angle0 + code_angle

        z0, z1, z2, z3 = heights[i]

        points.append([[radius * np.cos(angle0), radius * np.sin(angle0), z0],
                       [radius * np.cos(angle1), radius * np.sin(angle1), z1],
                       [radius * np.cos(angle1), radius * np.sin(angle1), z2],
                       [radius * np.cos(angle0), radius * np.sin(angle0), z3]])
    return np.array(points, dtype=np.float32)


def main(a, h):
    cap = cv2.VideoCapture(1)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    camera_matrix = np.array([[1000.0, 0.0, 500.0],
                              [0.0, 1000.0, 500.0],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    circumference = 225

    heights = [
        (h, h, h, h),
        (h, 0, 0, h),
        (0, 0, h, h),
        (0, h, h, 0),
        (h, h, 0, 0)
    ]


    all_obj_points = generate_ring_obj_points(circumference / (2 * np.pi), 5, a, heights)

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
    args = parser.parse_args()
    main(args.a, args.h)
