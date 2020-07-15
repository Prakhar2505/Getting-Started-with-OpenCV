import cv2
import numpy as numpy

image = cv2.imread('images/WaldoBeach.jpg')

cv2.imshow('Where is Waldo ?',image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

template = cv2.imread('images/waldo.jpg')

img_scaled = cv2.resize(template, None, fx=3.5, fy=3.5, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Find Waldo !',img_scaled)
cv2.waitKey(0)

grayTemp = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(gray,grayTemp,cv2.TM_CCOEFF)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Waldo Found', image)
cv2.waitKey(0)
cv2.destroyAllWindows()