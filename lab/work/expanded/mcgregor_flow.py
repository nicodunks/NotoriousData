"""Background camera-motion diagnostic; never a direct footwork or stance measure."""
import json,cv2,numpy as np,torch
from pathlib import Path
from ultralytics import YOLO
R=Path.cwd();O=R/'outputs/expanded/mcgregor/flow';O.mkdir(exist_ok=True);torch.set_num_threads(3);det=YOLO(str(R/'work/models/mma-detector/best.pt'))
for folder in ['pilot','expansion','expansion_extra']:
 base=R/'outputs/expanded/mcgregor'/folder
 for row in json.load(open(base/'manifest.json')):
  cid=row['anonymous_id'];dest=O/f'{cid}.json'
  if dest.exists():continue
  labpath=base/'labels'/f'{cid}.json'
  if not labpath.exists():continue
  lab=json.load(open(labpath))
  if lab['live_fight']!='yes' or lab['likely_replay']!='no' or lab['target_identity_visibility']!='high' or not any(p['phase']=='standing' for p in lab['phases']):continue
  cap=cv2.VideoCapture(str(R/f'work/raw/{row["source"]}.mp4'));images=[];masks=[]
  for j in range(32):
   cap.set(cv2.CAP_PROP_POS_MSEC,(row['start']+j/4)*1000);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,360));mask=np.full((360,640),255,np.uint8);mask[:30]=0;mask[320:]=0
   rr=det.predict(im,device='mps',conf=.35,imgsz=640,verbose=False)[0]
   for b in rr.boxes.xyxy.cpu().numpy():
    x1,y1,x2,y2=b.astype(int);cv2.rectangle(mask,(max(0,x1-12),max(0,y1-12)),(min(639,x2+12),min(359,y2+12)),0,-1)
   images.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY));masks.append(mask)
  pairs=[]
  for j in range(31):
   t=(j+1)/4
   if not any(p['start']<=t<p['end'] and p['phase']=='standing' for p in lab['phases']):continue
   mask=cv2.bitwise_and(masks[j],masks[j+1]);p=cv2.goodFeaturesToTrack(images[j],maxCorners=400,qualityLevel=.01,minDistance=8,mask=mask);reason=None;dx=dy=scale=rotation=None;inliers=0;tracked=0
   if p is None or len(p)<20:reason='insufficient detected-person-masked background features'
   else:
    q,status,error=cv2.calcOpticalFlowPyrLK(images[j],images[j+1],p,None,winSize=(21,21),maxLevel=3);sel=status.ravel().astype(bool)&(error.ravel()<25);p=p[sel].reshape(-1,2);q=q[sel].reshape(-1,2);tracked=len(p)
    if tracked<20:reason='insufficient low-error feature tracks'
    else:
     affine,inside=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99)
     if affine is None:reason='background similarity estimate failed'
     else:
      inliers=int(inside.sum());scale=float(np.hypot(affine[0,0],affine[1,0]));rotation=float(np.degrees(np.arctan2(affine[1,0],affine[0,0])));dx=float(affine[0,2]);dy=float(affine[1,2])
      if inliers<20 or inliers/tracked<.55 or not .9<scale<1.1 or abs(rotation)>8:reason='camera cut/perspective/model fit gate rejected'
   if any(abs(t-cut)<=.25 for cut in lab['camera']['cuts']):reason='reviewed camera-cut neighborhood'
   pairs.append({'time_start':j/4,'time_end':t,'accepted_background_model':reason is None,'reason':reason,'background_tracks':tracked,'inliers':inliers,'translation_pixels':[dx,dy],'scale':scale,'rotation_degrees':rotation})
  dest.write_text(json.dumps({'anonymous_id':cid,'source':row['source'],'method':'4Hz detected-person-masked background LK tracks + RANSAC similarity; reviewed cuts excluded','limitation':'Diagnostic of camera motion only. Cage/referee/crowd/missed detections and parallax may contaminate fit. No target foot displacement or bounce inferred from background fit.','pairs':pairs},indent=2));print(cid,flush=True)
