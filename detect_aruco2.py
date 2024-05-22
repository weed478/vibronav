import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def calculate_needle_base_distance(code_side_length: int, spacing: float, coef: float, distance: float) -> float:
    """Calculate the distance from the center of the central marker to the needle base."""
    
    if distance == 0:
        return 0
    
    half_code_size = code_side_length / 2

    # distance from the center of the central marker to the board edge
    edge_dist = half_code_size + spacing + coef * (code_side_length + 2 * spacing)

    # distance from the board edge to the needle base
    return edge_dist + distance


def get_marker_center_positions(code_side_length: float, spacing: float, coef: float) -> list:
    """Get positions of markers centers arranged in shape '+'."""

    half_code_size = code_side_length / 2

    positions = [
        (0, 0),                                             # central marker
        ((1 + coef) * (half_code_size + spacing), 0),       # right marker
        (0, (1 + coef) * (half_code_size + spacing)),       # top marker
        (-((1 + coef) * (half_code_size + spacing)), 0),    # left marker
        (0, -((1 + coef) * (half_code_size + spacing)))     # bottom marker
    ]
    return positions

def get_marker_corners_heights(board_height: float) -> list:
    """Get markers corners heights on a board."""

    h = board_height
    heights = [
        (h, h, h, h),
        (h, 0, 0, h),
        (0, 0, h, h),
        (0, h, h, 0),
        (h, h, 0, 0)
    ]
    return heights

def generate_markers_corners_coordinates(points: list, x: float, y: float, z_values: tuple, offset_x: float, offset_y: float) -> list:
    """Generate markers corners coordinates relative to markers centers. 
    Take into account the perceived distances caused by board design."""
    
    z0, z1, z2, z3 = z_values
    
    points.append([[x - offset_x, y + offset_y, z0],
                   [x + offset_x, y + offset_y, z1],
                   [x + offset_x, y - offset_y, z2],
                   [x - offset_x, y - offset_y, z3]])

def generate_plus_obj_points(code_side_length: int, board_height: float, spacing: float, distance: float) -> np.ndarray:
    """Generate coordinates of markers corners relative to markers centers."""

    if code_side_length == 0:
        raise ValueError("Code side length cannot be zero.")

    points = []

    # Perceived marker side length (seen from above)
    perceived_side_length = np.sqrt(code_side_length ** 2 - board_height ** 2)   
    
    # The scaling coefficient derived from the perceived side length and code size.
    coef = perceived_side_length / code_side_length

    base_dist = calculate_needle_base_distance(code_side_length, spacing, coef, distance)
    positions = get_marker_center_positions(code_side_length, spacing, coef)
    heights = get_marker_corners_heights(board_height)

    # offset variations derived from board design
    offset = code_side_length / 2
    scaled_offset = coef * offset

    for i in range(len(positions)):
        x, y = positions[i]

        # Shift on the X axis, so that the rendered axis overlaps the needle
        x -= base_dist       

        z_values = heights[i]

        if i == 0:
            generate_markers_corners_coordinates(points, x, y, z_values, offset, offset)

        if (i % 2) == 1:
            generate_markers_corners_coordinates(points, x, y, z_values, scaled_offset, offset)

        elif (i % 2) == 0 and i != 0:
            generate_markers_corners_coordinates(points, x, y, z_values, offset, scaled_offset)

    return np.array(points, dtype=np.float32)


def main(code_side_length, board_height, spacing, distance, needle_length):
    cap = cv2.VideoCapture(1)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    camera_matrix = np.array([[1000.0, 0.0, 500.0],
                              [0.0, 1000.0, 500.0],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    all_obj_points = generate_plus_obj_points(code_side_length, board_height, spacing, distance)

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
            # print('object points: ', obj_points)
            # print('image points: ', image_points)

            try:
                _, rvecs, tvecs = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, needle_length)
            except:
                pass

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to main')
    parser.add_argument('-a', '--code_side_length', type=int, help='Marker side length')
    parser.add_argument('-b', '--board_height', type=float, help='Board height')
    parser.add_argument('-s', '--spacing', type=float, help='Spacing between aruco markers')
    parser.add_argument('-d', '--distance', type=float, help='Board distance from the needle')
    parser.add_argument('-l', '--needle_length', type=float, help='Needle length')
    args = parser.parse_args()
    main(args.code_side_length, args.board_height, args.spacing, args.distance, args.needle_length)
