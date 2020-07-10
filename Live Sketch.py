import cv2
import numpy

cap = cv2.VideoCapture(0)


def sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(gaussian, 20, 30)
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)

    ret, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY_INV)

    return thresh, thresh2


while True:
    ret, frame = cap.read()
    out = sketch(frame)
    cv2.imshow('Sketch1', out[0])
    cv2.imshow('Sketch2', out[1])

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()