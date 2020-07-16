import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
image1 = image.copy()
image2 = image.copy()

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)


keypoints_sift, descriptors = sift.detectAndCompute(image, None)
keypoints_surf, descriptors = surf.detectAndCompute(image1, None)
keypoints_orb, descriptors = orb.detectAndCompute(image2, None)

image = cv2.drawKeypoints(img, keypoints_sift, None)
image1 = cv2.drawKeypoints(img, keypoints_surf, None)
image2 = cv2.drawKeypoints(img, keypoints_orb, None)



cv2.imshow("SIRF", image)
cv2.waitKey(0)


cv2.imshow("SURF", image1)
cv2.waitKey(0)


cv2.imshow("ORB", image2)
cv2.waitKey(0)


cv2.destroyAllWindows()