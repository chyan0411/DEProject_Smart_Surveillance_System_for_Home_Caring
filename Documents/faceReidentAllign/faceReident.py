import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import io
import imutils
import time
import numpy as np
import pickle
import scipy.spatial

# Load the model.
detectNet = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
reidentNet = cv.dnn.readNet('face-reidentification-retail-0095.xml',
                     'face-reidentification-retail-0095.bin')
# Specify target device.
detectNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
reidentNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# starting video streaming
cv.namedWindow('face')
vs = PiVideoStream().start()
time.sleep(2.0)

detectionThreshold = 0.6

scaleX = 456
scaleY = 342

#load the encoded faces
data = pickle.loads(open('/home/pi/Documents/faceReident/encodings.pickle', "rb").read())
print (data)

# load tensorflow model for emotion recognition

while True:
      startTime = time.time()
      
      frameArray = vs.read()

      captureTime = time.time()

      frameClone = imutils.resize(frameArray, width = 456)
      
      name = "unknown"

      # Prepare input blob and perform an inference.
      blob = cv.dnn.blobFromImage(frameArray, size=(300, 300), ddepth=cv.CV_8U)

      prepareTime = time.time()
      detectNet.setInput(blob)
      outputBlob = detectNet.forward()
      processTime = time.time()
      
      # Draw rectangles for detected faces
      
      faces = 0
      minDistance = 1
      index = 0
      
      for i in range(200):
            if outputBlob[0,0,i,2] > 0.3:
                  xMin = int(outputBlob[0,0,i,3] * scaleX)
                  yMin = int(outputBlob[0,0,i,4] * scaleY)
                  xMax = int(outputBlob[0,0,i,5] * scaleX)
                  yMax = int(outputBlob[0,0,i,6] * scaleY)
                  cv.rectangle(frameClone,(xMin, yMin), (xMax, yMax), (0,255,0),3)
                  faces += 1
      
      if faces > 0:
            roi = frameClone[yMin:yMax, xMin:xMax]
            if roi.any():
                  # compute the 256 value array for the detected face:
                  reidentBlob = cv.dnn.blobFromImage(roi, size=(128, 128), ddepth=cv.CV_8U)
                  reidentNet.setInput(reidentBlob)
                  reidentOutputBlob = reidentNet.forward()
                  
                  #compare the array to the ones in the database
                  for x in range (len(data["encodings"])-1):
                        distance = scipy.spatial.distance.cosine(reidentOutputBlob, data["encodings"][x])
                        if distance < minDistance:
                              minDistance = distance
                              index = x
                              
            if minDistance < detectionThreshold:
                  name = data["names"][index]
            cv.putText(frameClone, name, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv.LINE_AA)
            print (minDistance)

                        
                  
      cv.imshow('face', frameClone)
      
      #calculating the durations
      showTime = time.time()
      loopDuration =  '%.2f' % (showTime - startTime)
      captureDuration = '%.2f' % (captureTime - startTime)
      prepareDuration = '%.2f' % (prepareTime - captureTime)
      processDuration = '%.2f' % (processTime - prepareTime)
      showDuration = '%.2f' % (showTime - processTime)
      
      #write the durations into a file
      with open('/home/pi/Documents/faceReident/Durations.txt', 'a') as the_file:
              the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')
      

    

      if cv.waitKey(1) & 0xFF == ord('q'):
          break
                
                
cv.destroyAllWindows()
