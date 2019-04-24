from flask import Flask, render_template
from flask_socketio import SocketIO
import json
import base64
import datetime

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

import base64
from PIL import Image
import StringIO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

violenceDetector = ViolenceDetector()

count=0
baslangicsn=list()
bitissn=list()

@app.route('/')
def sessions():
   return render_template('index_webcam.html')

def messageReceived(methods=['GET', 'POST']):
   print('message was received!!!')

@socketio.on('DetectFrames')
def handle_my_custom_event(frames, methods=['GET', 'POST']):
   dataFrames=list()
   dataJson = json.loads(str(frames).replace('\'','\"'))
   for item in dataJson['data']:
      imgdata = base64.b64decode(item)
      dataFrames.append(imgdata)
      now = datetime.datetime.now()
      response={'isDone':'false','message':'Cevap zamani:'+str(now.isoformat())}
      socketio.emit('DetectFramesResponse', response, callback=messageReceived)
   print("Eleman sayisi:"+str(len(dataFrames)))
   now = datetime.datetime.now()
   response={'isDone':'true','message':'Cevap zamani:'+str(now.isoformat())}
   socketio.emit('DetectFramesResponse', response, callback=messageReceived)

@socketio.on('Detector')
def Detector(frames, methods=['GET', 'POST']):
   dataFrames=list()
   dataJson = json.loads(str(frames).replace('\'','\"'))
   for item in dataJson['data']:
      # print(item[item.index("base64,")+7:])
      imgdata = readb64(item)

      dataFrames.append(imgdata)
      # Capture frame-by-frame
      ret, currentImage = imgdata
      # do what you want with frame
      #  and then save to file
      count +=1
      netInput = ImageUtils.ConvertImageFrom_CV_to_NetInput(currentImage)
      startDetectTime = time.time()
      isFighting = violenceDetector.Detect(netInput)
      endDetectTime = time.time()
      

      targetSize = deploySettings.DISPLAY_IMAGE_SIZE - 2*deploySettings.BORDER_SIZE
      currentImage = cv2.resize(currentImage, (targetSize, targetSize))

      if isFighting:#şiddet tespit edildi
         if len(baslangicsn)==len(bitissn):
               baslangicsn.append(count/25)
         response={'isDone':'false','message':'Baslangic:'+str(baslangicsn.append(count/25))}
         socketio.emit('Detector', response, callback=messageReceived)

      else:
         if len(baslangicsn)!=len(bitissn):
               bitissn.append(count/25)
         response={'isDone':'false','message':'Bitis:'+str(bitissn.append(count/25))}
         socketio.emit('Detector', response, callback=messageReceived)
   print("Eleman sayisi:"+str(len(dataFrames)))
   bitissn.append(count/25)
   response={'isDone':'true','message':'tespit bitti'}
   socketio.emit('Detector', response, callback=messageReceived)

def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
   socketio.run(app, debug=True)