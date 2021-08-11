import cv2

image = cv2.imread('image6.jpg')



classNames = []
classFile = 'Labels.txt'
with open(classFile, 'rt') as f:
    classNames =  f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt '
weightsPath= 'frozen_inference_graph.pb'

img = cv2.dnn_DetectionModel(weightsPath,configPath)
img.setInputSize(700,800)
img.setInputScale(1.0/127.5)
img.setInputMean((127.5,127.5,127.5))
img.setInputSwapRB(True)

classIds, confs, bbox =  img.detect(image,confThreshold=0.5)
print(classIds, bbox)


for classId,confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(image,box,color=(0,0,0),thickness=3)
    cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)

cv2.imshow("Output",image)
cv2.waitKey(0)


