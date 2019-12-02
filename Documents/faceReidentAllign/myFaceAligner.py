import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import pickle
import scipy.spatial
import io
import imutils
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array

# Load the model.
faceNet = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
poseNet = cv.dnn.readNet('human-pose-estimation-0001.xml',
                     'human-pose-estimation-0001.bin')
reidentNet = cv.dnn.readNet('face-reidentification-retail-0095.xml',
                     'face-reidentification-retail-0095.bin')
# Specify target device.
faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
poseNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
reidentNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

#load the encoded faces
data = pickle.loads(open('/home/pi/Documents/faceReident/encodings.pickle', "rb").read())
detectionThreshold = 0.6

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
            frameCloneClean = imutils.resize(frameArray, width = 456)
            gray = cv.cvtColor(frameClone, cv.COLOR_BGR2GRAY)

            # Prepare input blob and perform an inference.
            faceBlob = cv.dnn.blobFromImage(frameArray, size=(300, 300), ddepth=cv.CV_8U)
            poseBlob = cv.dnn.blobFromImage(frameArray, size=(456, 256), ddepth=cv.CV_8U)

            prepareTime = time.time()
            
            faceNet.setInput(faceBlob)
            outputBlob = faceNet.forward()
            
            poseNet.setInput(poseBlob)
            heatmap = poseNet.forward()
			
            processTime = time.time()
            

            leftEyeX = 0
            leftEyeY = 0
            rightEyeX = 0
            rightEyeY = 0
            
            #draw pose
            
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
                    if i == 14:
                        rightEyeX = int(x*8)
                        rightEyeY = int(y*11)
                        
                    if i == 15:
                        leftEyeX = int(x*8)
                        leftEyeY = int(y*11)
                        
            #allign the face:
						
            dY = rightEyeY - leftEyeY
            dX = rightEyeX - leftEyeX
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = 0.3670542
            desiredDist *= 128
            scale = desiredDist / dist
            
            eyesCenter = ((leftEyeX + rightEyeX) // 2, (leftEyeY + rightEyeY) // 2)

            M = cv.getRotationMatrix2D(eyesCenter, angle, scale)
            
            tX = 128 * 0.5
            tY = 128 * 0.46157410714286
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
            
            output = cv.warpAffine(frameCloneClean, M, (128, 128), flags=cv.INTER_CUBIC)
            cv.imshow("Aligned", output)
            
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
                        
            name = "unknown"
            
            if faces > 0:
                  
                  roi = gray[yMin:yMax, xMin:xMax]
                  if roi.any():
                        roi = cv.resize(roi, (48, 48))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                    
                        preds = sess.run(tensor_output,  {tensor_input: roi})
                        label = EMOTIONS[preds.argmax()]
                        cv.putText(frameClone, label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv.LINE_AA)
                  
                  roi = output
                  if roi.any():
                      # compute the 256 value array for the detected face:
                      reidentBlob = cv.dnn.blobFromImage(roi, size=(128, 128), ddepth=cv.CV_8U)
                      #reidentBlob = cv.dnn.blobFromImage(frameCloneClean[yMin:yMax, xMin:xMax], size=(128, 128), ddepth=cv.CV_8U)
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
                      cv.putText(frameClone, name, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv.LINE_AA)
                        
            cv.imshow('face', frameClone)
            
            showTime = time.time()
            loopDuration =  '%.2f' % (showTime - startTime)
            captureDuration = '%.2f' % (captureTime - startTime)
            prepareDuration = '%.2f' % (prepareTime - captureTime)
            processDuration = '%.2f' % (processTime - prepareTime)
            showDuration = '%.2f' % (showTime - processTime)

            with open('/home/pi/Documents/faceReidentAllign/Durations.txt', 'a') as the_file:
                    the_file.write('Whole Loop: ' + loopDuration + 's\n'
                                    + 'Capture: ' + captureDuration + 's\n'
                                    + 'Prepare: ' + prepareDuration + 's\n'
                                    + 'Process: ' + processDuration + 's\n'
                                    + 'Show: ' + showDuration + 's\n\n')
          

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                
                
cv.destroyAllWindows()
