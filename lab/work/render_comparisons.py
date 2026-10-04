import json,subprocess,time,shutil
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
ROOT=Path.cwd();OUT=ROOT/'outputs';(OUT/'clips').mkdir(exist_ok=True);(OUT/'videos').mkdir(exist_ok=True);(OUT/'stills').mkdir(exist_ok=True)
S=json.load(open('work/samples.json'))
EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COLORS=[(235,195,100),(100,160,235),(170,235,130),(190,130,230)]
LABELS={'generic':'A / Generic YOLO11s pose','mma':'B / UFC-trained YOLO11x pose','rtmw':'C / UFC detector + RTMW-x'}

def draw(img,row):
 for idx,p in enumerate(row['people']):
  col=COLORS[idx%len(COLORS)];b=np.array(p['box']).astype(int);cv2.rectangle(img,tuple(b[:2]),tuple(b[2:]),col,2)
  k=np.array(p['keypoints'])
  def line(a,b,width=2):
   if a<len(k) and b<len(k) and k[a,2]>=.3 and k[b,2]>=.3:cv2.line(img,tuple(k[a,:2].astype(int)),tuple(k[b,:2].astype(int)),col,width,cv2.LINE_AA)
  for a,b in EDGES:line(a,b)
  if len(k)>100:
   for root in (91,112):
    for j in range(5):
     chain=[root]+list(range(root+1+j*4,root+5+j*4))
     for a,b in zip(chain,chain[1:]):line(a,b,1)
  for j,(x,y,c) in enumerate(k):
   if c>=.3 and (j<23 or j>=91):cv2.circle(img,(int(x),int(y)),2 if j<23 else 1,col,-1,cv2.LINE_AA)
 return img

def phase_at(sid,t):
 p=OUT/'phases'/f'{sid}.json'
 if not p.exists():return 'Phase pending'
 d=json.load(open(p));rows=d.get('windows',d.get('results',[]))
 for r in rows:
  a=r.get('start_s',r.get('start',0));b=r.get('end_s',a+5)
  if a<=t<b:
   return ('Non-fight' if r.get('excluded') else str(r.get('phase',r.get('phase_label','unknown'))))+' / 5s window'
 return 'Phase: unknown'

for s in S:
 shutil.copy2(s['path'],OUT/'clips'/f'{s["id"]}.mp4')
 data={key:json.load(open(OUT/'pose'/f'{s["id"]}__{key}.json')) for key in LABELS}
 cap=cv2.VideoCapture(s['path']);fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4));out=OUT/'videos'/f'{s["id"]}-comparison.mp4'
 ff=imageio_ffmpeg.get_ffmpeg_exe();proc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x840','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
 i=0
 while True:
  ok,f=cap.read()
  if not ok:break
  canvas=np.full((840,1280,3),(46,29,20),np.uint8)
  for slot,key in enumerate(['original','generic','mma','rtmw']):
   x=(slot%2)*640;y=(slot//2)*420;im=f.copy();n=0
   if key!='original':
    row=data[key]['frames'][i];im=draw(im,row);n=len(row['people'])
   label='Reference / Original footage' if key=='original' else LABELS[key]
   cv2.putText(canvas,label,(x+14,y+25),cv2.FONT_HERSHEY_SIMPLEX,.63,(244,234,218),1,cv2.LINE_AA)
   text=f'{s["start"]+i/fps:.2f}s source / {i/fps:.2f}s clip' if key=='original' else f'{n} detections / joint threshold 0.30'
   cv2.putText(canvas,text,(x+14,y+47),cv2.FONT_HERSHEY_SIMPLEX,.43,(192,177,153),1,cv2.LINE_AA)
   canvas[y+60:y+420,x:x+640]=cv2.resize(im,(640,360))
  proc.stdin.write(canvas.tobytes())
  if i in (12,96,180):cv2.imwrite(str(OUT/'stills'/f'{s["id"]}-{i}.jpg'),canvas)
  i+=1
 proc.stdin.close();rc=proc.wait();assert rc==0
 print(out,i,flush=True)
