##K.K.R.R.Premathilaka
##INTERNET OF THINGS AND COMPUTER VISION
##Task 01: Object Detection / Optical Character Recognition (ORC)


import cv2

image = cv2.imread('image2.jpg')



classNames = []
classFile = 'Labels.txt'
with open(classFile, 'rt') as f:
    classNames =  f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt '
weightsPath= 'frozen_inference_graph.pb'

img = cv2.dnn_DetectionModel(weightsPath,configPath)
img.setInputSize(1200,1000)
img.setInputScale(1.0/127.5)
img.setInputMean((127.5,127.5,127.5))
img.setInputSwapRB(True)

classIds, confs, bbox =  img.detect(image,confThreshold=0.5)
print(classIds, bbox)


for classId,confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(image,box,color=(255,255,255),thickness=3)
    cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)

cv2.imshow("Output",image)
cv2.waitKey(0)


