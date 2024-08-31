import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageStitcher_SIFT:
    """
    A simple class for image stitching using SIFT features and RANSAC.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matching_method = cv2.BFMatcher()
        self.ransac_threshold = 5.0
        self.ransac_reproj_threshold = 5.0

    def detect_and_compute(self, image):
        """
        Detect and compute SIFT keypoints and descriptors.

        Parameters:
        image (np.ndarray): Input image.

        Returns:
        tuple: Keypoints and descriptors.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_keypoints(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """
        Match keypoints between two images.

        Parameters:
        keypoints1 (list): Keypoints of the first image.
        descriptors1 (np.ndarray): Descriptors of the first image.
        keypoints2 (list): Keypoints of the second image.
        descriptors2 (np.ndarray): Descriptors of the second image.

        Returns:
        list: Matches between keypoints.
        """
        matches = self.matching_method.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        """
        Estimate the homography matrix using RANSAC.

        Parameters:
        keypoints1 (list): Keypoints of the first image.
        keypoints2 (list): Keypoints of the second image.
        matches (list): Matches between keypoints.

        Returns:
        tuple: Homography matrix and inliers.
        """
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, self.ransac_reproj_threshold)
        return M, mask

    def stitch_images(self, image1, image2):
        """
        Stitch two images together.

        Parameters:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.

        Returns:
        np.ndarray: Stitched image.
        """
        keypoints1, descriptors1 = self.detect_and_compute(image1)
        keypoints2, descriptors2 = self.detect_and_compute(image2)

        matches = self.match_keypoints(keypoints1, descriptors1, keypoints2, descriptors2)

        M, _ = self.estimate_homography(keypoints1, keypoints2, matches)

        result_width = image1.shape[1] + image2.shape[1]
        result_height = max(image1.shape[0], image2.shape[0])
        result = cv2.warpPerspective(image1, M, (result_width, result_height))
        result[0:image2.shape[0], 0:image2.shape[1]] = image2

        return result

    def display_stitched_image(self, image):
        """
        Display the stitched image.

        Parameters:
        image (np.ndarray): Stitched image.
        """
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Stitched Image')
        plt.axis('off')
        plt.show()
class ImageStitcher_ORB(ImageStitcher_SIFT):
    def __init__(self):
        # Initialize the ORB detector with a reasonable number of keypoints
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Create a brute force matcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def stitch(self, img1, img2, draw_matches=False):
        # Detect and compute features for both images
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        # Match descriptors.
        matches = self.bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x:x.distance)

        # Select good matches
        good_matches = matches[:100]  # Select top 100 matches

        if draw_matches:
            # Draw first 100 matches.
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return img_matches

        # Extract the matched keypoints coordinates
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to img2's plane based on the homography
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Warp img1 using the homography matrix M
        warped_img1 = cv2.warpPerspective(img1, M, (w2, h2))

        # Combine the two images
        result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        result[:h2, :w2] = img2
        result[:h1, w2:w1+w2] = warped_img1

        # Fill in the overlapping region
        mask = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32([ [w2, 0], [w1+w2, 0], [w1+w2, h1], [w2, h1] ]), (255))
        result = cv2.bitwise_and(result, result, mask=mask)

        return result