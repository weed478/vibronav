import cv2
import cv2.aruco as aruco
import numpy as np


def generate_grid_obj_points(rows, cols, code_size, spacing):
    points = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * (code_size + spacing)
            x1 = x0 + code_size
            y0 = row * (code_size + spacing)
            y1 = y0 + code_size
            points.append([x0, y0, 0])
            points.append([x1, y0, 0])
            points.append([x1, y1, 0])
            points.append([x0, y1, 0])
    return np.array(points, dtype=np.float32)


def generate_ring_obj_points(radius, num_codes, code_size):
    circumference = 2 * np.pi * radius
    code_angle = code_size / circumference * 2 * np.pi
    points = []
    for i in range(num_codes):
        angle0 = 2 * np.pi * i / num_codes
        angle1 = angle0 + code_angle
        points.append([[radius * np.cos(angle0), radius * np.sin(angle0), 0],
                       [radius * np.cos(angle1), radius * np.sin(angle1), 0],
                       [radius * np.cos(angle1), radius * np.sin(angle1), code_size],
                       [radius * np.cos(angle0), radius * np.sin(angle0), code_size]])
    return np.array(points, dtype=np.float32)


def main():
    cap = cv2.VideoCapture(1)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    camera_matrix = np.array([[1000.0, 0.0, 500.0],
                              [0.0, 1000.0, 500.0],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    circumference = 225
    all_obj_points = generate_ring_obj_points(circumference / (2 * np.pi), 18, 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            obj_points = []
            image_points = []
            for id in range(24):
                if id not in ids:
                    continue
                i = np.where(ids == id)[0][0]
                obj_points.extend(all_obj_points[id])
                image_points.extend(corners[i][0])
            obj_points = np.array(obj_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            _, rvecs, tvecs = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 50)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
