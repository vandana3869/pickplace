import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    
    #yellow color
    low_yellow = np.array([150,150,0])
    high_yellow=np.array([255,255,0])
    yellow_mask= cv2.inRange(hsv_frame, low_yellow, high_yellow)
    _, contours, _= cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    
    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)
        
        x_medium= int((x+x+w)/2)
        break
    
    cv2.imshow("Frame",frame)
    cv2.imshow("mask",yellow_mask)
               
    key = cv2.waitKey(1)
               
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()