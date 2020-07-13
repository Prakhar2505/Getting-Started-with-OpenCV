import numpy as np
import cv2

carCascade = cv2.CascadeClassifier('/home/crawler/Python Project/Getting-Started-with-OpenCV/Cascades/haarcascade_car.xml')

cap = cv2.VideoCapture('images/cars.avi')
cap.set(3,640)
cap.set(4,480)

while True:
    ret,image =cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    car = carCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for x,y,w,h in car:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Car Detection',image)

    if cv2.waitKey(30) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()    