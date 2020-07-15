import cv2
import numpy as np 

image = cv2.imread('images/chess.jpg')
image2 = image.copy()

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

corners = cv2.cornerHarris(gray,3,3,0.05)

kernel = np.ones((7,7),np.uint8)

corners = cv2.dilate(corners, kernel,iterations=2)

image[corners > 0.025 * corners.max() ] = [255, 0, 0]

cv2.imshow('Harris Corners', image)
cv2.waitKey(0)

corners2 = cv2.goodFeaturesToTrack(gray,100,0.01,1,50)

for corner in corners2:
    x, y = corner[0]
    x= int(x)
    y=int(y)
    cv2.rectangle(image2,(x-10,y-10),(x+10,y+10),(255,0,255),2)

cv2.imshow('Good Features to Track',image2)

cv2.waitKey()

cv2.destroyAllWindows()