import cv2
import numpy as np

PedDetect = cv2.CascadeClassifier('/home/crawler/Python Project/Getting-Started-with-OpenCV/Cascades/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('images/walking.avi')

while True:
    ret,image = cap.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    pedestrian = PedDetect.detectMultiScale(gray,1.05,3)

    for x,y,w,h in pedestrian:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('Pedestrian Detection',image)

    if cv2.waitKey(30)==27:
        break
cap.release()
cv2.destroyAllWindows()    