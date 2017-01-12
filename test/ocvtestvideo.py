import os.path
import time
import numpy as np
import cv2
import Tkinter
import tkFileDialog


cap = cv2.VideoCapture("/home/ivusi7dl/Videos/Leauto_20161013_201432A.MP4")
#cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret==True):
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	#cv2.imshow('frame',gray)
    	cv2.imshow('frame',frame)
    	if cv2.waitKey(10) & 0xFF == ord('q'):
    		break
    else:
    	break
       

cap.release()
cv2.destroyAllWindows()
