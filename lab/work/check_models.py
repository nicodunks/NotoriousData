import cv2,time
from ultralytics import YOLO
from rtmlib import RTMPose
from pathlib import Path
p='work/raw/pereira-adesanya.mp4';c=cv2.VideoCapture(p);c.set(cv2.CAP_PROP_POS_MSEC,255000);ok,f=c.read();assert ok
for name,path in [('generic','yolo11s-pose.pt'),('mma','work/models/mma-pose/best.pt'),('detector','work/models/mma-detector/best.pt')]:
 m=YOLO(path);t=time.perf_counter();r=m.predict(f,device='mps',verbose=False,conf=.25)[0];print(name,len(r.boxes),time.perf_counter()-t,flush=True);cv2.imwrite('work/check-'+name+'.jpg',r.plot())
url='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'
p=RTMPose(url,model_input_size=(288,384),backend='onnxruntime',device='cpu');print('RTMW loaded',flush=True)
