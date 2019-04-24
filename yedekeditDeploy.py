#!/usr/bin/python3

import os
import sys
import cv2
import numpy as np
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

def PrintHelp():
	print("Usage:")
	print("\t $(ThisScript)  $(PATH_FILE_NAME_OF_SOURCE_VIDEO)")
	print()
	print("or, specified $(PATH_FILE_NAME_TO_SAVE_RESULT) to save detection result:")
	print("\t $(ThisScript)  $(PATH_FILE_NAME_OF_SOURCE_VIDEO)  $(PATH_FILE_NAME_TO_SAVE_RESULT)")
	print()

class VideoSavor:
	def AppendFrame(self, image_):
		self.outputStream.write(image_)

	def __init__(self, targetFileName, videoCapture):
		width = int( deploySettings.DISPLAY_IMAGE_SIZE )
		height = int( deploySettings.DISPLAY_IMAGE_SIZE )
		frameRate = int( videoCapture.get(cv2.CAP_PROP_FPS) )
		codec = cv2.VideoWriter_fourcc(*'XVID')
		self.outputStream = cv2.VideoWriter(targetFileName + ".avi", codec, frameRate, (width, height) )
		

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


def DetectViolence(PATH_FILE_NAME_OF_SOURCE_VIDEO, PATH_FILE_NAME_TO_SAVE_RESULT):
	
	violenceDetector = ViolenceDetector()
	videoReader = cv2.VideoCapture(PATH_FILE_NAME_OF_SOURCE_VIDEO)
	shouldSaveResult = (PATH_FILE_NAME_TO_SAVE_RESULT != None)

	if shouldSaveResult:
		videoSavor = VideoSavor(PATH_FILE_NAME_TO_SAVE_RESULT + "_Result", videoReader)

	listOfForwardTime = []
	isCurrentFrameValid, currentImage = videoReader.read()
	data = DataSet()
	model = load_model('/home/furkan/five-video-classification-methods-master/inception.023-3.04.hdf5')
	# Predict.
	image_arr = np.expand_dims(currentImage, axis=0)
	predictions = model.predict(image_arr)

	label_predictions = {}
	for i, label in enumerate(data.classes):
		label_predictions[label] = predictions[0][i]
	
	sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
	listeString = list()
	listeValue =  list()
	for i, class_prediction in enumerate(sorted_lps):
			# Just get the top five.
		if i > 4:
			break
		#print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
		listeString.append(class_prediction[0])
		listeValue.append(class_prediction[1])
		maxValue = max(listeValue)
		maxValueIndex = listeValue.index(maxValue)
		#print(maxValueIndex,"--",maxValue)
		#print(listeString[maxValueIndex])

		i += 1


	X=0
	while isCurrentFrameValid:
		netInput = ImageUtils.ConvertImageFrom_CV_to_NetInput(currentImage)
		startDetectTime = time.time()
		isFighting = violenceDetector.Detect(netInput)
		endDetectTime = time.time()
		listOfForwardTime.append(endDetectTime - startDetectTime)
		
		


		targetSize = deploySettings.DISPLAY_IMAGE_SIZE - 2*deploySettings.BORDER_SIZE
		currentImage = cv2.resize(currentImage, (targetSize, targetSize))
		

		if isFighting:#şiddet tespit edildi
			
			if X == 50:
				listeString.clear()
				listeValue.clear()
				image_arr = np.expand_dims(currentImage, axis=0)
				predictions = model.predict(image_arr)
				
				label_predictions = {}
				for i, label in enumerate(data.classes):
					label_predictions[label] = predictions[0][i]
	
				sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

				for i, class_prediction in enumerate(sorted_lps):
					# Just get the top five.
					if i > 4:
						break
					#print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
					listeString.append(class_prediction[0])
					listeValue.append(class_prediction[1])
					maxValue = 0
					maxValue = max(listeValue)
					maxValueIndex = listeValue.index(maxValue)
					print(listeString[maxValueIndex],"--",maxValue)
					print(listeString[maxValueIndex])

					i += 1
				x=0
			
			else:			
				X +=1
				
			resultImage = cv2.copyMakeBorder(currentImage,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 cv2.BORDER_CONSTANT,
							 value=deploySettings.FIGHT_BORDER_COLOR)
			font = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (10,300)
			fontScale = 1
			fontColor = (255,255,255)
			lineType = 2

			cv2.putText(resultImage,listeString[maxValueIndex],bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
			print(listeString[maxValueIndex],"--",maxValue)


		else:
			resultImage = cv2.copyMakeBorder(currentImage,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 deploySettings.BORDER_SIZE,
							 cv2.BORDER_CONSTANT,
							 value=deploySettings.NO_FIGHT_BORDER_COLOR)
			


		cv2.imshow("Violence Detection", resultImage)
		if shouldSaveResult:
			videoSavor.AppendFrame(resultImage)

		userResponse = cv2.waitKey(1)
		if userResponse == ord('q'):
			videoReader.release()
			cv2.destroyAllWindows()
			break

		else:
			isCurrentFrameValid, currentImage = videoReader.read()

	PrintUnsmoothedResults(violenceDetector.unsmoothedResults)
	averagedForwardTime = np.mean(listOfForwardTime)
	print("Averaged Forward Time: ", averagedForwardTime)
	


if __name__ == '__main__':
	if len(sys.argv) >= 2:
		PATH_FILE_NAME_OF_SOURCE_VIDEO = sys.argv[1]

		try:
			PATH_FILE_NAME_TO_SAVE_RESULT = sys.argv[2]
		except:
			PATH_FILE_NAME_TO_SAVE_RESULT = None

		if os.path.isfile(PATH_FILE_NAME_OF_SOURCE_VIDEO):
			DetectViolence(PATH_FILE_NAME_OF_SOURCE_VIDEO, PATH_FILE_NAME_TO_SAVE_RESULT)

		#else:
			#raise ValueError("Not such file: " + videoPathName)

	else:
		PrintHelp()
