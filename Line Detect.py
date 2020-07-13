import cv2 
import numpy as np
img = cv2.imread('images/soduku.jpg')
image=img.copy()
image2=img.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,170,apertureSize=3)

cv2.imshow('Edges',edges)
cv2.waitKey(0)

lines = cv2.HoughLines(edges,1,np.pi/180,240)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow('Hough Line',image)
cv2.waitKey(0)

lines = cv2.HoughLinesP(edges,1,np.pi/180,200,5,10)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Probabilistic Hough Line',image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


