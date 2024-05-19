import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def generate_plus_obj_points(code_size: int, plate_height: float, spacing: float, distance: float) -> np.ndarray:
    """Generate coordinates of markers corners relative to markers centers."""

    points = []

    half_code_size = code_size / 2
    b = np.sqrt(code_size**2 - plate_height**2)   # code side length seen from above
    coef = b / code_size

    base_dist = 0
    if distance:
        edge_dist = half_code_size + spacing + b + coef * (2 * spacing)        # distance from center of the central marker to plate edge
        base_dist = edge_dist + distance                                       # distance from plate edge to needle base

    # positions of centers of markers in shape "+"
    positions = [
        (0, 0),                                                                    # central marker
        ((half_code_size + spacing) + coef * (half_code_size + spacing), 0),       # right marker
        (0, (half_code_size + spacing) + coef * (half_code_size + spacing)),       # top marker
        (-((half_code_size + spacing) + coef * (half_code_size + spacing)), 0),    # left marker
        (0, -((half_code_size + spacing) + coef * (half_code_size + spacing)))     # bottom marker
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

    # generate corner coordinates relative to markers centers

    # central marker (zero)
    x, y = positions[0]
    z0, z1, z2, z3 = heights[0]
    points.append([[x - half_code_size - base_dist, y + half_code_size, z0],
                    [x + half_code_size - base_dist, y + half_code_size, z1],
                    [x + half_code_size - base_dist, y - half_code_size, z2],
                    [x - half_code_size - base_dist, y - half_code_size, z3]])
    
    # marker id=1
    x, y = positions[1]
    z0, z1, z2, z3 = heights[1]
    points.append([[x - coef * half_code_size - base_dist, y + half_code_size, z0],
                    [x + coef * half_code_size - base_dist, y + half_code_size, z1],
                    [x + coef * half_code_size - base_dist, y - half_code_size, z2],
                    [x - coef * half_code_size - base_dist, y - half_code_size, z3]])
    
    # marker id=2
    x, y = positions[2]
    z0, z1, z2, z3 = heights[2]
    points.append([[x - half_code_size - base_dist, y + coef * half_code_size, z0],
                    [x + half_code_size - base_dist, y + coef * half_code_size, z1],
                    [x + half_code_size - base_dist, y - coef * half_code_size, z2],
                    [x - half_code_size - base_dist, y - coef * half_code_size, z3]])
    
    # marker id=3
    x, y = positions[3]
    z0, z1, z2, z3 = heights[3]
    points.append([[x - coef * half_code_size - base_dist, y + half_code_size, z0],
                    [x + coef * half_code_size - base_dist, y + half_code_size, z1],
                    [x + coef * half_code_size - base_dist, y - half_code_size, z2],
                    [x - coef * half_code_size - base_dist, y - half_code_size, z3]])
    
    # marker id=4
    x, y = positions[4]
    z0, z1, z2, z3 = heights[4]
    points.append([[x - half_code_size - base_dist, y + coef * half_code_size, z0],
                    [x + half_code_size - base_dist, y + coef * half_code_size, z1],
                    [x + half_code_size - base_dist, y - coef * half_code_size, z2],
                    [x - half_code_size - base_dist, y - coef * half_code_size, z3]])

    return np.array(points, dtype=np.float32)


def main(a, h, s, d, l):
    cap = cv2.VideoCapture(1)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    camera_matrix = np.array([[1000.0, 0.0, 500.0],
                              [0.0, 1000.0, 500.0],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    all_obj_points = generate_plus_obj_points(a, h, s, d)

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
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, l)
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
    parser.add_argument('d', type=float, help='Plate distance from the needle')
    parser.add_argument('l', type=float, help='Needle length')
    args = parser.parse_args()
    main(args.a, args.h, args.s, args.d, args.l)
