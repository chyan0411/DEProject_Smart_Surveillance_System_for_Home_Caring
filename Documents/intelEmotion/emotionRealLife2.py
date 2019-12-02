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
net = cv.dnn.readNet('emotions-recognition-retail-0003.xml',
                     'emotions-recognition-retail-0003.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# starting video streaming
cv.namedWindow('emotion')
vs = PiVideoStream().start()
time.sleep(2.0)


while True:
      startTime = time.time()
      
      frameArray = vs.read()

      captureTime = time.time()

      frameClone = imutils.resize(frameArray, width = 456)

      # Prepare input blob and perform an inference.
      blob = cv.dnn.blobFromImage(frameArray, size=(456, 256), ddepth=cv.CV_8U)

      prepareTime = time.time()
      net.setInput(blob)
      heatmap = net.forward()
      processTime = time.time()
      
      # Draw detected pose on the frame.

      for i in range(18):
              v = 0
              x = 0
              y = 0
              for j in range(32):
                  for k in range(57):
                      if heatmap[0,i,j,k] > v:
                          v = heatmap[0,i,j,k]
                          x = k
                          y = j
              
              if v > 0.4:
                  cv.circle(frameClone,(int(x*8) ,int(y*11)), 5, (0,0,255), -1)
                  
                  
      cv.imshow('pose', frameClone)
      
      showTime = time.time()
      loopDuration =  '%.2f' % (showTime - startTime)
      captureDuration = '%.2f' % (captureTime - startTime)
      prepareDuration = '%.2f' % (prepareTime - captureTime)
      processDuration = '%.2f' % (processTime - prepareTime)
      showDuration = '%.2f' % (showTime - processTime)

      with open('/home/pi/Documents/intelPose/Durations2.txt', 'a') as the_file:
              the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')
    

      if cv.waitKey(1) & 0xFF == ord('q'):
          break
                
                
cv.destroyAllWindows()
