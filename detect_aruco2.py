import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def calculate_needle_base_distance(code_side_length: int, spacing: float, coef: float, distance: float) -> float:
    """
    Calculate the distance from the center of the central marker to the needle base.

    Parameters:
    code_side_length (int): The length of the code side.
    spacing (float): The spacing between markers.
    coef (float): The scaling coefficient.
    distance (float): The distance from the board edge to the needle base.

    Returns:
    float: The calculated distance from the center of the central marker to the needle base.

    Notes:
    If the distance is 0, the function returns 0. Otherwise, it calculates the distance from the center of the
    central marker to the board edge, adds the distance from the board edge to the needle base, and returns
    the result.
    """
    
    if distance == 0:
        return 0
    
    half_code_size = code_side_length / 2

    # Distance from the center of the central marker to the board edge
    edge_dist = half_code_size + spacing + coef * (code_side_length + 2 * spacing)

    # Distance from the board edge to the needle base
    return edge_dist + distance


def get_marker_center_positions(code_side_length: float, spacing: float, coef: float) -> list:
    """
    Get the positions of markers centers arranged in the shape of '+'.

    Parameters:
    code_side_length (float): The length of the code side.
    spacing (float): The spacing between markers.
    coef (float): The scaling coefficient.

    Returns:
    list: A list containing tuples representing the positions of markers centers.
          Each tuple contains two elements (x, y) representing the coordinates of
          a marker center. 

    Notes:
    - The positions of markers centers are calculated taking into account the board design
    and the need for distance scaling.
    - The order of elements in the list corresponds to the arrangement of markers. 
    The order corresponds to their IDs: 0 (central), 1 (right), 2 (top), 3 (left), 4 (bottom).
    """

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
    """
    Get the heights of marker corners relative to the board.

    Parameters:
    board_height (float): The height of the protrusion of board.

    Returns:
    list: A list containing tuples representing the heights of marker corners.
          Each tuple contains four elements corresponding to the heights of the
          four corners of a marker. The order of heights in each tuple is (z0, z1, z2, z3).
        
    Representations:
    z0: The height of the top-left corner of the marker.
    z1: The height of the top-right corner of the marker.
    z2: The height of the bottom-right corner of the marker.
    z3: The height of the bottom-left corner of the marker.
    """

    h = board_height
    heights = [
        (h, h, h, h),
        (h, 0, 0, h),
        (0, 0, h, h),
        (0, h, h, 0),
        (h, h, 0, 0)
    ]
    return heights

def generate_markers_corners_coordinates(points: list, x: float, y: float, z_values: tuple, offset_x: float, offset_y: float) -> None:
    """
    Generate the coordinates of marker corners relative to their centers.

    Parameters:
    points (list): A list to append the generated corner coordinates.
    x (float): The x-coordinate of the marker center.
    y (float): The y-coordinate of the marker center.
    z_values (tuple): A tuple containing the z-coordinates for the four corners of the marker (z0, z1, z2, z3).
    offset_x (float): The offset to apply along the x-axis for the corner coordinates.
    offset_y (float): The offset to apply along the y-axis for the corner coordinates.


    The function calculates the coordinates of the corners of a marker based on the 
    center coordinates (x, y), applying the specified offsets along the x and y axes.
    The z-coordinates are taken from the provided tuple, allowing for a 3D representation 
    of the marker corners.
    """
    
    z0, z1, z2, z3 = z_values
    
    points.append([[x - offset_x, y + offset_y, z0],
                   [x + offset_x, y + offset_y, z1],
                   [x + offset_x, y - offset_y, z2],
                   [x - offset_x, y - offset_y, z3]])

def generate_plus_obj_points(code_side_length: int, board_height: float, spacing: float, distance: float) -> np.ndarray:
    """
    Generate coordinates of the corners of markers arranged in a "+" shape, relative to their centers.

    Parameters:
    code_side_length (int): The side length of the code marker.
    board_height (float): The height of the board protrusion on which markers are placed.
    spacing (float): The spacing between the ArUco markers.
    distance (float): The distance from the board edge to the needle base.

    Returns:
    np.ndarray: An array containing the coordinates of the corners of the markers.

    Raises:
    ValueError: If code_side_length is zero.

    The function calculates the perceived side length of the markers seen from above,
    the base distance from the board edge to the needle base, and the positions of the 
    marker centers. It then generates the coordinates of the corners of each marker 
    relative to their centers, adjusting for the board design and spacing.
    """

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

    # Offset variations derived from board design
    offset = code_side_length / 2
    scaled_offset = coef * offset

    for i in range(len(positions)):
        x, y = positions[i]

        # Shift on the X axis, so that the rendered axis overlaps the needle
        x -= base_dist       

        z_values = heights[i]

        if i == 0:
            generate_markers_corners_coordinates(points, x, y, z_values, offset, offset)
        elif (i % 2) == 1:
            generate_markers_corners_coordinates(points, x, y, z_values, scaled_offset, offset)
        else:
            generate_markers_corners_coordinates(points, x, y, z_values, offset, scaled_offset)

    return np.array(points, dtype=np.float32)


def main(code_side_length, board_height, spacing, distance, needle_length):
    """
    Execute the main functionality of the program.

    Args:
        code_side_length (int): The length of the side of the marker code.
        board_height (float): The height of the board protrusion.
        spacing (float): The spacing between markers.
        distance (float): The distance from the board to the needle.
        needle_length (float): The length of the needle.

    Returns:
        None

    Notes:
        - This function captures video from the specified camera and detects markers using the specified parameters.
        - It then calculates the position and orientation of the needle relative to the markers.
        - The calculated needle position and orientation are visualized on the video feed.
    """
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
    parser.add_argument('-b', '--board_height', type=float, help='Board protrusion height')
    parser.add_argument('-s', '--spacing', type=float, help='Spacing between aruco markers')
    parser.add_argument('-d', '--distance', type=float, help='Board distance from the needle')
    parser.add_argument('-l', '--needle_length', type=float, help='Needle length')
    args = parser.parse_args()
    main(args.code_side_length, args.board_height, args.spacing, args.distance, args.needle_length)
