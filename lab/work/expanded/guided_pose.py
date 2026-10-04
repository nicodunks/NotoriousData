import os
import json,time,hashlib
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from rtmlib import RTMPose
R=Path.cwd();O=R/'outputs/expanded/identity';checkpoint=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'))
model=RTMPose(str(checkpoint),model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1;model.session=ort.InferenceSession(str(checkpoint),sess_options=opts,providers=['CPUExecutionProvider'])
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for rt in (91,112):
 for j in range(5):
  ch=[rt]+list(range(rt+1+j*4,rt+5+j*4));EDGES+=list(zip(ch,ch[1:]))
for sid in 'CD':
 path=O/'guided_audit'/f'{sid}.json'
 while not path.exists():time.sleep(3)
 # Wait for metadata and parseable final output.
 while True:
  try:guide=json.load(open(path));assert 'frames' in guide;break
  except:time.sleep(3)
 data=json.load(open(R/f'outputs/experiment/body_data/{sid}_rtmw_133.json'));cap=cv2.VideoCapture(data['source']['path']);results=[];tiles=[]
 for row in guide['frames']:
  fr=row['frame'];cap.set(cv2.CAP_PROP_POS_MSEC,fr/data['fps']*1000);ok,im=cap.read();fighters=[p for p in row['fighters'] if p['box_normalized'] is not None];boxes=[]
  for p in fighters:
   b=np.array(p['box_normalized'])*[960,540,960,540];b[[0,2]]=np.clip(b[[0,2]],0,959);b[[1,3]]=np.clip(b[[1,3]],0,539);boxes.append(b.tolist())
  xy,scores=model(im,boxes) if boxes else ([],[]);ps=[]
  for p,b,k,sc in zip(fighters,boxes,xy,scores):
   key=p['identity'];col=(238,211,34) if key=='fighter_black' else (133,113,251);kp=np.c_[k,sc];ps.append({**p,'box':b,'keypoints':kp.round(3).tolist(),'display_color':'#22d3ee' if key=='fighter_black' else '#fb7185','keypoint_ownership':'guided crop candidate; unverified under overlap','score_units':'raw SIMCC peak; not probability'})
   cv2.rectangle(im,tuple(np.array(b[:2]).astype(int)),tuple(np.array(b[2:]).astype(int)),col,2);cv2.putText(im,key+' / guided estimate',tuple((np.array(b[:2])+[0,18]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1)
   valid=(kp[:,0]>=0)&(kp[:,0]<960)&(kp[:,1]>=0)&(kp[:,1]<540)&(kp[:,2]>=2)
   for a,z in EDGES:
    if valid[a] and valid[z]:cv2.line(im,tuple(kp[a,:2].astype(int)),tuple(kp[z,:2].astype(int)),col,1,cv2.LINE_AA)
   for j in np.where(valid)[0]:cv2.circle(im,tuple(kp[j,:2].astype(int)),1,col,-1)
  results.append({'frame':fr,'time':fr/data['fps'],'people':ps});cv2.putText(im,f'{sid} {fr/data["fps"]:.2f}s - GUIDED CANDIDATES, ownership unverified',(5,525),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);tiles.append(cv2.resize(im,(640,360)))
 cv2.imwrite(str(O/f'{sid}-guided-contact.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]));(O/f'{sid}-guided-poses.json').write_text(json.dumps({'source':data['source'],'fps':data['fps'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'method':'tool-free Sol per-fighter box guidance then separate RTMW inference; no training','limitations':['Guided boxes overlap and contain opponent limbs.','Separate pose inference may still put both skeletons on the same fighter.','No calibrated joint visibility, no inferred contact truth.','Sparse sampled frames only; no dense interpolated identity claims.'],'frames':results},separators=(',',':')));print('GUIDED',sid,flush=True)
