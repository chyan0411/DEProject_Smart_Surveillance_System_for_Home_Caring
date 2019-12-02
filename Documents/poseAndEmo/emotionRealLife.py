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
import tensorflow as tf
from keras.preprocessing.image import img_to_array

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
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# load tensorflow model for emotion recognition
with tf.Session() as sess:
      with tf.gfile.FastGFile('/home/pi/Documents/tfEmotion/tf_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
    
      tensor_output = sess.graph.get_tensor_by_name('import/global_average_pooling2d_1/Mean:0')
      tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')


      while True:
            startTime = time.time()
            
            frameArray = vs.read()

            captureTime = time.time()

            frameClone = imutils.resize(frameArray, width = 456)

            # Prepare input blob and perform an inference.
            blob = cv.dnn.blobFromImage(frameArray, size=(300, 300), ddepth=cv.CV_8U)

            prepareTime = time.time()
            net.setInput(blob)
            outputBlob = net.forward()
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
            
            if faces > 0:
                  gray = cv.cvtColor(frameClone, cv.COLOR_BGR2GRAY)
                  roi = gray[yMin:yMax, xMin:xMax]
                  if roi.any():
                        roi = cv.resize(roi, (48, 48))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                    
                        preds = sess.run(tensor_output,  {tensor_input: roi})
                        label = EMOTIONS[preds.argmax()]
                        print (label)
                        
            cv.imshow('face', frameClone)
            
            showTime = time.time()
            loopDuration =  '%.2f' % (showTime - startTime)
            captureDuration = '%.2f' % (captureTime - startTime)
            prepareDuration = '%.2f' % (prepareTime - captureTime)
            processDuration = '%.2f' % (processTime - prepareTime)
            showDuration = '%.2f' % (showTime - processTime)

            with open('/home/pi/Documents/tfEmotion/Durations.txt', 'a') as the_file:
                    the_file.write('Whole Loop: ' + loopDuration + 's\n'
                                    + 'Capture: ' + captureDuration + 's\n'
                                    + 'Prepare: ' + prepareDuration + 's\n'
                                    + 'Process: ' + processDuration + 's\n'
                                    + 'Show: ' + showDuration + 's\n\n')
          

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                
                
cv.destroyAllWindows()
