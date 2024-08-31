from stich import ImageStitcher_SIFT, ImageStitcher_ORB
import cv2
import time
import os


def get_filename(file_path):
    return os.path.splitext(os.path.split(file_path)[-1])[0]


stitcher = ImageStitcher_ORB()
p1 = 'images/DSC01188.jpg'
p2 = 'images/DSC01189.jpg'
n1=get_filename((p1))
n2=get_filename((p2))
print(n1, n2)
# Load images
image1 = cv2.imread(p1)
image2 = cv2.imread(p2)
# im2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
# im1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

# Stitch images
start = time.time()

stitched_image = stitcher.stitch(image1, image2)
end = time.time()
print('elipsed:', end - start)
# Display the result
cv2.imwrite(f'stiched_image/{n1}_{n2}.jpg', stitched_image)
