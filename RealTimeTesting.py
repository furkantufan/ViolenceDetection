import numpy as np
import cv2
import os
import sys
import time
from src.ViolenceDetector import *
import settings.DeploySettings as deploySettings
import settings.DataSettings as dataSettings
import src.data.ImageUtils as ImageUtils
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model



cap = cv2.VideoCapture(0)
violenceDetector = ViolenceDetector()

count=0
baslangicsn=list()
bitissn=list()
	
	
while(True):
    # Capture frame-by-frame
    ret, currentImage = cap.read()
    # do what you want with frame
    #  and then save to file
    cv2.imwrite('/home/furkan/Desktop/image.png', currentImage)
    count +=1
    netInput = ImageUtils.ConvertImageFrom_CV_to_NetInput(currentImage)
    startDetectTime = time.time()
    isFighting = violenceDetector.Detect(netInput)
    endDetectTime = time.time()
    

    targetSize = deploySettings.DISPLAY_IMAGE_SIZE - 2*deploySettings.BORDER_SIZE
    currentImage = cv2.resize(currentImage, (targetSize, targetSize))

    if isFighting:#şiddet tespit edildi
        p=0
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,50)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        if len(baslangicsn)==len(bitissn):
            baslangicsn.append(count/25)

        cv2.putText(currentImage,"Siddet tespit edildi",bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
        bottomLeftCornerOfText = (10,450)

    else:

        if len(baslangicsn)!=len(bitissn):
            bitissn.append(count/25)
		
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,450)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(currentImage,"Siddet tespit edilmedi",bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
    
    if cv2.waitKey(30) & 0xFF == ord('q'): # you can increase delay to 2 seconds here
        break



bitissn.append(count/25)
print(len(baslangicsn),"-------",len(bitissn))
for index in range(len(baslangicsn)):
    try:
        print("tespit edilen sureler",baslangicsn.pop(index),"------",bitissn.pop(index))
    except IndexError:
        print ("----son----")




def PrintUnsmoothedResults(unsmoothedResults_):
	print("Unsmoothed results:")
	print("\t [ ")
	print("\t   ", end='')
	for i, eachResult in enumerate(unsmoothedResults_):
		if i % 10 == 9:
			print( str(eachResult)+", " )
			print("\t   ", end='')

		else:
			print( str(eachResult)+", ", end='')

	print("\n\t ]")
	

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
