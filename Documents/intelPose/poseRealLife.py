import cv2 as cv

import picamera
import picamera.array
import io
import imutils
import time
from imutils.video.pivideostream import PiVideoStream
from imutils.video import VideoStream


# Load the model.
net = cv.dnn.readNet('human-pose-estimation-0001.xml',
                     'human-pose-estimation-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# starting video streaming
cv.namedWindow('your_face')
#vs = PiVideoStream().start()
vs = VideoStream(usePiCamera = True).start()
time.sleep(2.0)

while True:
      startTime = time.time()

      frameArray = vs.read()
      

      captureTime = time.time()

      #frameProcessed = imutils.resize(frameArray, width = 456)
      #frameClone = frameProcessed.copy()


      # Prepare input blob and perform an inference.
      blob = cv.dnn.blobFromImage(frameArray, size=(456, 256), ddepth=cv.CV_8U)

      prepareTime = time.time()
      net.setInput(blob)
      heatmap = net.forward()

      processTime = time.time()
      # Draw detected pose on the frame.
      print (heatmap.shape)

      #frame = cv.resize(frame, (456, 256))

      for i in range(19):
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
                  cv.circle(frameArray,(int(x*8) ,int(y*8)), 5, (0,0,255), -1)
                  



      cv.imshow('your_face', frameArray)
      #cv.waitKey(0)
      showTime = time.time()
      loopDuration =  '%.2f' % (showTime - startTime)
      captureDuration = '%.2f' % (captureTime - startTime)
      prepareDuration = '%.2f' % (prepareTime - captureTime)
      processDuration = '%.2f' % (processTime - prepareTime)
      showDuration = '%.2f' % (showTime - processTime)

      with open('/home/pi/Documents/intelPose/Durations.txt', 'a') as the_file:
              the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')


      if cv.waitKey(1) & 0xFF == ord('q'):
          break
                
                
cv.destroyAllWindows()
