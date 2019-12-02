import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import io
import imutils
import time

# Load the model.
net = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

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
      net.setInput(blob)
      outputBlob = net.forward()
      processTime = time.time()
      
      # Draw rectangles for detected faces
      
      print(outputBlob.shape)
      for i in range(200):
            if outputBlob[0,0,i,2] > 0.3:
                  print('gesicht ist da')
                  print(outputBlob[0,0,i,3])
                  print(outputBlob[0,0,i,4])
                  print(outputBlob[0,0,i,5])
                  print(outputBlob[0,0,i,6])
                  xMin = int(outputBlob[0,0,i,3] * scaleX)
                  yMin = int(outputBlob[0,0,i,4] * scaleY)
                  xMax = int(outputBlob[0,0,i,5] * scaleX)
                  yMax = int(outputBlob[0,0,i,6] * scaleY)
                  cv.rectangle(frameClone,(xMin, yMin), (xMax, yMax), (0,255,0),3)
                  
      cv.imshow('face', frameClone)
      
      showTime = time.time()
      loopDuration =  '%.2f' % (showTime - startTime)
      captureDuration = '%.2f' % (captureTime - startTime)
      prepareDuration = '%.2f' % (prepareTime - captureTime)
      processDuration = '%.2f' % (processTime - prepareTime)
      showDuration = '%.2f' % (showTime - processTime)

      with open('/home/pi/Documents/intelFace/Durations.txt', 'a') as the_file:
              the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')
    

      if cv.waitKey(1) & 0xFF == ord('q'):
          break
                
                
cv.destroyAllWindows()
