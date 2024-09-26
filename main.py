import cv2
#from djitellopy import tello
import cvzone

thres=0.6
nmsThres=0.2

cap=cv2.VideoCapture(0)
# setting the size
cap.set(3,640)
cap.set(4,480)

classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames =f.read().split('\n')
print(classNames)

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthsPath="frozen_inference_graph.pb"

net=cv2.dnn_DetectionModel(weigthsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img=cap.read()
    classIds,confs,bbox=net.detect(img,confThreshold=thres,nmsThreshold=nmsThres)

    try:
        for classId,confs,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(classIds,{classNames[classId - 1].upper()},confs,box)
            cvzone.cornerRect(img,box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round((confs * 100), 2)}',
                        (box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
    except:
        pass
    cv2.imshow("Image",img)
    cv2.waitKey(1)

