import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import io
import imutils
import numpy
import time
from keras.preprocessing.image import img_to_array


# Load the models
faceNet = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
                     
emotionNet = cv.dnn.readNet('emotions-recognition-retail-0003.xml',
                        'emotions-recognition-retail-0003.bin')
                        
emotionLabels = ["neutral", "happy", "sad", "surprise", "anger"]

# Specify target device.
faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
emotionNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# starting video streaming
cv.namedWindow('face')
vs = PiVideoStream().start()
time.sleep(2.0)

scaleX = 456
scaleY = 342

while True:
      startTime = time.time()
      
      frameArray = vs.read()

      captureTime = time.time()

      frameClone = imutils.resize(frameArray, width = 456)
      #print (frameClone.shape)

      # Prepare input blob and perform an inference.
      blob = cv.dnn.blobFromImage(frameArray, size=(300, 300), ddepth=cv.CV_8U)

      prepareTime = time.time()
      faceNet.setInput(blob)
      outputBlob = faceNet.forward()
      processTime = time.time()
      
      # Draw rectangles for detected faces
      
      faces = 0

      for i in range(200):
            if outputBlob[0,0,i,2] > 0.3:
                  
                  xMin = int(outputBlob[0,0,i,3] * scaleX)
                  yMin = int(outputBlob[0,0,i,4] * scaleY)
                  xMax = int(outputBlob[0,0,i,5] * scaleX)
                  yMax = int(outputBlob[0,0,i,6] * scaleY)
                  cv.rectangle(frameClone,(xMin, yMin), (xMax, yMax), (0,255,0),3)
                  
                  faces += 1
      
                  
                  
      #cut only the face
      if faces > 0:
            gray = cv.cvtColor(frameClone, cv2.COLOR_BGR2GRAY)
            roi = gray[yMin:yMax, xMin:xMax]
            roi = cv.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            #img_to_arry is function from keras!!!
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

      
      cv.imshow('face', frameClone)
      
      showTime = time.time()
      loopDuration =  '%.2f' % (showTime - startTime)
      captureDuration = '%.2f' % (captureTime - startTime)
      prepareDuration = '%.2f' % (prepareTime - captureTime)
      processDuration = '%.2f' % (processTime - prepareTime)
      showDuration = '%.2f' % (showTime - processTime)

      with open('/home/pi/Documents/intelEmotion/Durations.txt', 'a') as the_file:
              the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')
    

      if cv.waitKey(1) & 0xFF == ord('q'):
          break
                
                
cv.destroyAllWindows()
