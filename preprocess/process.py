import numpy as np
import cv2
import matplotlib.pyplot as plt


def cut_into_array(arr, top_left, bottom_right):
    """
    Cut a specific region into the array with values from a specified position.

    Parameters:
    arr (np.ndarray): The input array.
    top_left (tuple): Top-left corner coordinates.
    bottom_right (tuple): Bottom-right corner coordinates.
    """
    arr[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = arr[top_left[0] - 3, top_left[1] - 10]


def find_position(binary_image):
    """
    Find the position of the first non-zero element in the binary image.

    Parameters:
    binary_image (np.ndarray): Binary image.

    Returns:
    tuple or bool: Position as (row, column) or False if no non-zero elements.
    """
    non_zero_rows = np.any(binary_image, axis=1)
    if not non_zero_rows.any():
        return False
    first_non_zero_row_index = np.where(non_zero_rows)[0][0]
    n = binary_image[first_non_zero_row_index]
    non_zero_cols = np.nonzero(n)[0]
    col = non_zero_cols[0]
    row = first_non_zero_row_index
    return row, col


def find_corner_tips(image, block_size, aperture_size, threshold):
    """
    Detect corner tips in an image using the Harris corner detector.

    Parameters:
    image (np.ndarray): Input image.
    block_size (int): Size of neighborhood considered for corner detection.
    aperture_size (int): Aperture parameter for Sobel operator.
    threshold (float): Threshold value for corner detection.

    Returns:
    np.ndarray: Binary image indicating corners.
    """
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, block_size, aperture_size, 0.20)
    im = np.zeros(gray.shape)
    im[dst > threshold * dst.max()] = 1
    return im


def preprocess_image(image, top_left=False, bottom_right=False):
    """
    Preprocess an image by converting it to grayscale, blurring, and edge detection.

    Parameters:
    image (np.ndarray): Input image.
    top_left (tuple): Top-left corner coordinates for cropping.
    bottom_right (tuple): Bottom-right corner coordinates for cropping.

    Returns:
    np.ndarray: Processed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=255)
    if top_left is not False and bottom_right is not False:
        edges[:top_left[0], :] = 0
        edges[:, :top_left[1]] = 0
        edges[bottom_right[0]:, :] = 0
        edges[:, bottom_right[1]:] = 0
    return edges


def find_curve_points(images, top_left, bottom_right):
    """
    Find curve points in a list of images.

    Parameters:
    images (list): List of image file paths.
    top_left (tuple): Top-left corner coordinates for cropping.
    bottom_right (tuple): Bottom-right corner coordinates for cropping.

    Returns:
    tuple: Two lists containing points and indices.
    """
    points = []
    indices = []
    in_region = False
    point_list = []
    index_list = []

    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        processed_img = preprocess_image(img, top_left, bottom_right)
        tip = find_corner_tips(processed_img, 3, 5, 0.01)
        p = find_position(tip)
        if p and in_rectangle(p[0],p[1]):
            in_region = True
            points.append(p)
            indices.append(i)
        elif in_region:
            in_region = False
            point_list.append(points[:])
            index_list.append(indices[:])
            points.clear()
            indices.clear()

    return point_list, index_list


def in_rectangle(x, y):
    """
    Check if a point is within a specified rectangle.

    Parameters:
    x (int): X-coordinate.
    y (int): Y-coordinate.

    Returns:
    bool: True if the point is inside the rectangle, False otherwise.
    """
    return 590 <= y <= 660
