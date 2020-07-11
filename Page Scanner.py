import cv2
import numpy as np 
image =cv2.imread('images/scan.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
cv2.waitKey()
cv2.destroyAllWindows()
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(gaussian, 30, 50)
contour,hierarchy = cv2.findContours(canny.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour, key = cv2.contourArea, reverse = True)
for cnt in contour :
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

    if len(approx)==4 :
        target = approx
        break

target = target.reshape((4,2))
hnew = np.zeros((4,2),dtype = np.float32)

add = target.sum(1)
hnew[0] = target[np.argmin(add)]
hnew[2] = target[np.argmax(add)]

diff = np.diff(target,axis = 1)
hnew[1] = target[np.argmin(diff)]
hnew[3] = target[np.argmax(diff)]
pts=np.float32([[0,0],[800,0],[800,1000],[0,1000]])  

op=cv2.getPerspectiveTransform(hnew,pts) 
dst=cv2.warpPerspective(image,op,(800,1000))


cv2.imshow("Scanned",dst)
cv2.waitKey()
cv2.destroyAllWindows()