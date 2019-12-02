from imutils import paths
import pickle
import cv2 as cv
import os
import imutils
import sys

#load the models
detectNet = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
reidentNet = cv.dnn.readNet('face-reidentification-retail-0095.xml',
                     'face-reidentification-retail-0095.bin')
detectNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
reidentNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
scaleX = 456
scaleY = 342


def encodeFaces():
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    #imagePaths = list(paths.list_images('/home/pi/Documents/faceReident/datasets'))
    imagePaths = list(paths.list_images(os.path.join(sys.path[0], 'dataset/')))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv.imread(imagePath)
        
        frameClone = imutils.resize(image, width = 456)
        
        blob = cv.dnn.blobFromImage(frameClone, size=(300, 300), ddepth=cv.CV_8U)
        detectNet.setInput(blob)
        outputBlob = detectNet.forward()
        
        faces = 0
        
        for i in range(200):
            if outputBlob[0,0,i,2] > 0.3:
                xMin = int(outputBlob[0,0,i,3] * scaleX)
                yMin = int(outputBlob[0,0,i,4] * scaleY)
                xMax = int(outputBlob[0,0,i,5] * scaleX)
                yMax = int(outputBlob[0,0,i,6] * scaleY)
                faces += 1
          
        if faces > 0:
            roi = frameClone[yMin:yMax, xMin:xMax]
            if roi.any():
                reidentBlob = cv.dnn.blobFromImage(roi, size=(128, 128), ddepth=cv.CV_8U)
                reidentNet.setInput(reidentBlob)
                reidentOutputBlob = reidentNet.forward()
                      
                print (reidentOutputBlob.shape)
                print (name)
                knownEncodings.append(reidentOutputBlob)
                knownNames.append(name)


    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    #f = open('/home/pi/Documents/faceReident/encodings.pickle', "wb")
    f = open(os.path.join(sys.path[0], 'encodings.pickle'), "wb")
    f.write(pickle.dumps(data))
    f.close()
