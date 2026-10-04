import os,time,json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from rtmlib import RTMPose
ROOT=Path.cwd(); OUT=ROOT/'outputs/pose';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6)
samples=json.load(open('work/samples.json'))
model_specs=[('generic','work/models/yolo11s-pose.pt'),('mma','work/models/mma-pose/best.pt'),('rtmw','work/models/mma-detector/best.pt')]
rtmw=None
for key,weight in model_specs:
 model=YOLO(weight)
 if key=='rtmw':
  import onnxruntime as ort
  p=os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')
  rtmw=RTMPose(p,model_input_size=(288,384),backend='onnxruntime',device='cpu')
  opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
  rtmw.session=ort.InferenceSession(p,sess_options=opts,providers=['CPUExecutionProvider'])
 for s in samples:
  out=OUT/f'{s["id"]}__{key}.json'
  if out.exists():continue
  cap=cv2.VideoCapture(s['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4));rows=[];t0=time.perf_counter()
  for i in range(n):
   ok,f=cap.read()
   if not ok:break
   r=model.predict(f,device='mps',conf=.25,imgsz=640,verbose=False)[0]
   boxes=r.boxes.xyxy.cpu().numpy();conf=r.boxes.conf.cpu().numpy();people=[]
   if key=='rtmw':
    xy,sc=rtmw(f,boxes.tolist()) if len(boxes) else ([],[])
    kp=[np.c_[a,b].round(3).tolist() for a,b in zip(xy,sc)]
   else:
    kp=r.keypoints.data.cpu().numpy().round(3).tolist() if r.keypoints is not None else []
   for j,b in enumerate(boxes):
    people.append(dict(box=b.round(2).tolist(),score=round(float(conf[j]),4),keypoints=kp[j] if j<len(kp) else []))
   rows.append(dict(frame=i,time=round(i/fps,6),people=people))
   if i%60==0:print(key,s['id'],i,n,round(time.perf_counter()-t0,1),flush=True)
  elapsed=time.perf_counter()-t0
  data=dict(model=key,checkpoint=weight,checkpoint_sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),fps=fps,width=w,height=h,frames=rows,seconds=elapsed,threshold=.25,keypoint_threshold=.3,device='mps detector; cpu RTMW' if key=='rtmw' else 'mps',source=s)
  out.write_text(json.dumps(data,separators=(',',':')));print('DONE',out,round(elapsed,1),flush=True)
