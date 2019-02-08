import numpy as np
import cv2
import math

cap = cv2.VideoCapture(1)

def crop_minAreaRect(img, rect):
    
    # Let cnt be the contour and img be the input

    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))

    return croppedRotated

while(True):
    ret, frame = cap.read()
    frame = frame[60:-60, 0:-1]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    th3 = cv2.adaptiveThreshold(gray ,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    contours, _ = cv2.findContours(th3, 1, 2)

    biggestContour = []
    biggestArea = 0;
    biggestRect = None

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        hi1 = box[0][1] - box[1][1]
        len1 = box[0][0] - box[1][0]
        side1 = math.sqrt((hi1 * hi1) + (len1 * len1))

        hi2 = box[1][1] - box[2][1]
        len2 = box[1][0] - box[2][0]
        side2 = math.sqrt((hi2 * hi2) + (len2 * len2))

        area = side2 * side1

        if area > 10000 and area < (len(frame) * len(frame[0])) - 5000 and area > biggestArea:
            biggestArea = area
            biggestContour = box
            biggestRect = rect
    try:
        cv2.drawContours(frame,[biggestContour],0,(0,0,255),2)
    except:
        1
    cv2.imshow('frame', frame)
    try:
        cropped = crop_minAreaRect(frame, biggestRect)
        cv2.imshow('contours', cropped)
    except:
        1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()