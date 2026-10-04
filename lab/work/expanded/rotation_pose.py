import os
import json,hashlib,subprocess,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from rtmlib import RTMPose
R=Path.cwd();O=R/'outputs/expanded/identity';A=O/'rotation_audit';A.mkdir(exist_ok=True)
cp=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'));m=RTMPose(str(cp),model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1;m.session=ort.InferenceSession(str(cp),sess_options=opts,providers=['CPUExecutionProvider'])
d=json.load(open(O/'C-guided-poses.json'));cap=cv2.VideoCapture(d['source']['path']);records=[];images=[]
E=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
for row in [d['frames'][0],d['frames'][1]]:
 cap.set(cv2.CAP_PROP_POS_MSEC,row['time']*1000);ok,im=cap.read();tiles=[]
 for p in row['people']:
  x1,y1,x2,y2=np.array(p['box']).astype(int);crop=im[y1:y2,x1:x2];h,w=crop.shape[:2];group=[]
  for rotation in range(4):
   rotated=np.ascontiguousarray(np.rot90(crop,rotation));hh,ww=rotated.shape[:2];xy,sc=m(rotated,[[0,0,ww-1,hh-1]]);xy=xy[0];sc=sc[0];x,y=xy[:,0].copy(),xy[:,1].copy()
   if rotation==1:xy[:,0]=w-1-y;xy[:,1]=x
   elif rotation==2:xy[:,0]=w-1-x;xy[:,1]=h-1-y
   elif rotation==3:xy[:,0]=y;xy[:,1]=h-1-x
   xy += [x1,y1];kp=np.c_[xy,sc];records.append({'time':row['time'],'identity':p['identity'],'rotation_ccw_degrees':rotation*90,'box':p['box'],'keypoints':kp.round(3).tolist(),'ownership':'unverified hypothesis'})
   canvas=im.copy();col=(238,211,34) if p['identity']=='fighter_black' else (133,113,251)
   cv2.rectangle(canvas,(x1,y1),(x2,y2),col,1)
   for a,b in E:
    if min(sc[a],sc[b])>2:cv2.line(canvas,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),col,2)
   for j in list(range(23))+list(range(23,91))+[91,112]:
    if sc[j]>2:cv2.circle(canvas,tuple(xy[j].astype(int)),2,col,-1)
   for j,label in [(0,'nose'),(9,'LW'),(10,'RW'),(11,'LH'),(12,'RH')]:cv2.putText(canvas,label,tuple(xy[j].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
   cv2.putText(canvas,f'{row["time"]:.1f}s {p["identity"]} input rotation {rotation*90} CCW',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,col,2);group.append(cv2.resize(canvas,(640,360)))
  tiles+=group
 sheet=O/f'rotation-C-{int(row["time"]):02d}.jpg';cv2.imwrite(str(sheet),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)]));images.append(sheet)
 ref=O/f'rotation-C-{int(row["time"]):02d}-reference.jpg';cv2.imwrite(str(ref),im);images.append(ref)
(O/'rotation-C-hypotheses.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'method':'separate guided crops rotated 0/90/180/270 CCW, RTMW133, inverse coordinate transform','coordinate_units':'original 960x540 pixels; raw scores not probabilities','hypotheses':records},separators=(',',':')))
prompt='Use ONLY provided images. No tools, web, files, or fight memory. Images show original pixels and four RTMW rotation hypotheses for each of two clothing identities at times 0 and 4s. Cyan fighter_black has light hair and black shorts; coral fighter_red has red shorts. For EACH timestamp and identity, inspect face/nose, left/right wrists, hips against ORIGINAL visible anatomy. Choose rotation per body part only if clearly visibly better; null if hidden or mixed opponent ownership. Do NOT pick by model score or make an occluded point ground truth. Return JSON {"assessments":[{"time":0,"identity":"fighter_black","face":{"preferred_rotation":null,"evidence":"..."},"wrists":{"preferred_rotation":null,"evidence":"..."},"hips":{"preferred_rotation":null,"evidence":"..."},"failures":[]}],"overall":"Does alignment help? Describe visible improvements and failures without gold truth."}. All four time/identity assessments required.'
(A/'selection.prompt.txt').write_text(prompt);dest=A/'selection.json';log=A/'selection.jsonl';cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(R/'work/experiment/isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
for im in images:cmd+=['-i',str(im)]
cmd+=['-'];start=time.time()
with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=420)
ev=[]
for line in log.read_text().splitlines():
 try:ev.append(json.loads(line))
 except:pass
used=[e for e in ev if e.get('item',{}).get('type') not in [None,'agent_message','reasoning']];assert r.returncode==0 and not used
s=dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();dest.write_text(json.dumps(json.loads(s),indent=2));(A/'selection.metadata.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','tools_used':used,'seconds':time.time()-start,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'images':[{'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in images]},indent=2));print('DONE',flush=True)
