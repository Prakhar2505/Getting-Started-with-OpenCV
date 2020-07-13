import cv2
import numpy as np

image = cv2.imread('images/bottlecaps.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 20, minRadius = 32, maxRadius = 40)

for i in circles[0,:]:
       cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)
       cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)

cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
