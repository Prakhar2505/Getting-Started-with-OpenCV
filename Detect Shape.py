import cv2
import numpy as np 
image =cv2.imread('images/someshapes.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,1)
cv2.imshow("Thresholded image",thresh)
cv2.waitKey()
contour,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

for cnt in contour :
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

    if len(approx)==3 :
        shape = "Triangle"
    elif len(approx)==4 :
        x,y,w,h=cv2.boundingRect(cnt)
        if(abs(w-h)<=10):
            shape = "Square"
        else :
            shape = "Rectangle"
    elif len(approx)==10 :
        shape = "Star"
    elif len(approx)>10 :
        shape = "Circle"

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.drawContours(image,[cnt],0,(0,255,0),-1)
    cv2.putText(image,shape,(cx-60,cy),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)        
cv2.imshow("Processed",image)
cv2.waitKey()
cv2.destroyAllWindows()
