import os
import json,cv2,numpy as np,torch,onnxruntime as ort,hashlib
from rtmlib import RTMPose
from ultralytics import YOLO
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/pilot';(O/'pose_candidates').mkdir(exist_ok=True)
detpath=R/'work/models/mma-detector/best.pt';cp=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'));torch.set_num_threads(4);det=YOLO(str(detpath));m=RTMPose(str(cp),model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;m.session=ort.InferenceSession(str(cp),sess_options=opts,providers=['CPUExecutionProvider'])
E=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
for row in json.load(open(O/'manifest.json')):
 frames=[];tiles=[];cid=row['anonymous_id']
 for j in (0,16,31):
  path=O/'native'/f'{cid}-{j:02d}.jpg';im=cv2.imread(str(path));h,w=im.shape[:2];r=det.predict(im,device='mps',conf=.4,imgsz=960,verbose=False)[0];boxes=r.boxes.xyxy.cpu().numpy();xy,sc=m(im,boxes.tolist()) if len(boxes) else ([],[]);people=[]
  for n,(b,k,s) in enumerate(zip(boxes,xy,sc)):
   kp=np.c_[k,s];people.append({'detection_index':n,'box':b.round(2).tolist(),'keypoints':kp.round(3).tolist(),'identity':'unverified','joint_visibility':'unverified; raw scores not probabilities'})
   col=(238,211,34) if n%2==0 else (133,113,251);cv2.rectangle(im,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),col,2);cv2.putText(im,f'D{n} identity unverified',tuple((b[:2]+[0,25]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,col,2)
   for a,z in E:
    if min(s[a],s[z])>2:cv2.line(im,tuple(k[a].astype(int)),tuple(k[z].astype(int)),col,2)
   for q in range(23):
    if s[q]>2:cv2.circle(im,tuple(k[q].astype(int)),3,col,-1)
  frames.append({'time':j/4,'source_time':row['start']+j/4,'people':people});tile=cv2.resize(im,(960,540));cv2.putText(tile,f'{cid} t={j/4:.2f}s: candidate pose only',(8,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);tiles.append(tile)
 data={'anonymous_id':cid,'source':row['source'],'start':row['start'],'width':w,'height':h,'detector_sha256':hashlib.sha256(detpath.read_bytes()).hexdigest(),'rtmw_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'frames':frames,'limitations':['No identity from detection order','No calibrated visibility scores','No masked pixels','Pose candidates cannot establish weight distribution or world stance width']};(O/'pose_candidates'/f'{cid}.json').write_text(json.dumps(data,separators=(',',':')));cv2.imwrite(str(O/'pose_candidates'/f'{cid}.jpg'),np.vstack(tiles));print(cid,flush=True)
