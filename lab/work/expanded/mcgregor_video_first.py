import os
import json,cv2,numpy as np,torch,onnxruntime as ort,sys,concurrent.futures,subprocess,hashlib,time
from pathlib import Path
from rtmlib import RTMPose
from ultralytics import YOLO
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
R=Path.cwd();O=R/'outputs/expanded/mcgregor/tracked';O.mkdir(exist_ok=True)
for d in ['body_data','identity/anchor_audit','identity/audit','videos']:(O/d).mkdir(parents=True,exist_ok=True)
CASES={'P':('2012','Navy shorts, yellow gloves, red tape','White patterned shorts, yellow gloves, blue tape'),'Q':('2013','Black/green shorts, red tape','White patterned shorts, blue tape'),'R':('2018','Green shorts, blue tape','Black shorts, red tape'),'S':('2021','Green/olive shorts, blue tape','Black shorts, red tape')}
cp=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'));detpath=R/'work/models/mma-detector/best.pt';torch.set_num_threads(6);det=YOLO(str(detpath));m=RTMPose(str(cp),model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1;m.session=ort.InferenceSession(str(cp),sess_options=opts,providers=['CPUExecutionProvider'])
for cid,(year,target,other) in CASES.items():
 out=O/'body_data'/f'{cid}_rtmw_133.json';path=R/'outputs/expanded/mcgregor/four_bout/clips'/f'{year}-native-standing.mp4'
 if out.exists():continue
 c=cv2.VideoCapture(str(path));fps=c.get(5);n=int(c.get(7));rows=[]
 for seq,t in enumerate(np.arange(0,min(n/fps,8),1/6)):
  ix=min(n-1,round(t*fps));c.set(1,ix);ok,im=c.read();assert ok;im=im[:540];a=det.predict(im,device='mps',conf=.3,imgsz=960,verbose=False)[0];boxes=a.boxes.xyxy.cpu().numpy();xy,sc=m(im,boxes.tolist()) if len(boxes) else ([],[]);people=[{'box':b.round(2).tolist(),'keypoints':np.c_[k,s].round(3).tolist()} for b,k,s in zip(boxes,xy,sc)];rows.append({'frame':seq,'time':ix/fps,'native_frame':ix,'people':people})
  if seq%12==0:print('POSE',year,seq,flush=True)
 c.release();out.write_text(json.dumps({'model':'RTMW133','fps':6,'width':960,'height':540,'source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'frames':rows},separators=(',',':')))
 print('DONE POSE',year,flush=True)
sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as b;b.O=O/'identity'
def anchor(cid):
 year,target,other=CASES[cid];data=json.load(open(O/'body_data'/f'{cid}_rtmw_133.json'));cap=cv2.VideoCapture(data['source']['path']);tiles=[];indices=list(range(0,len(data['frames']),6))
 for ix in indices:
  row=data['frames'][ix];cap.set(1,row['native_frame']);ok,im=cap.read();assert ok;im=im[:540]
  for j,p in enumerate(row['people']):
   x1,y1,x2,y2=map(int,p['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(255,255,255),2);cv2.putText(im,'D'+str(j),(x1,max(22,y1+22)),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  im=cv2.resize(im,(640,360));cv2.putText(im,f'sample{ix} t={row["time"]:.2f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);tiles.append(im)
 cap.release();sheet=O/'identity/anchor_audit'/f'{cid}.jpg';cv2.imwrite(str(sheet),np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,len(tiles),2)]));prompt='Use only supplied numbered image sheet; no tools,files,web or fight memory. Classify each detector box clothing role. fighter_cyan='+target+'; fighter_coral='+other+'. If clothing differs, referee/background, identity is unclear or box merges both bodies, identity null. Do not force two identities. Return JSON {frames:[{frame:sampleindex,detections:[{detection_index:number,identity:fighter_cyan|fighter_coral|null,certainty:high|medium|low,evidence:string}]}],limitations:[]}. Sample indices '+str(indices)+'. Color assignment does not validate hidden joint anatomy.';a=b.call(cid+('_clothing_v2' if cid in 'PQ' else '_clothing'),prompt,[sheet]);(O/'identity/anchor_audit'/f'{cid}.json').write_text(json.dumps(a,indent=2));print('ANCHOR',year,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(anchor,CASES))
names={cid:{'fighter_cyan':v[1],'fighter_coral':v[2]} for cid,v in CASES.items()};s=(R/'work/expanded/identity.py').read_text();start=s.index('PALETTE=');end=s.index('EDGES=');s=s[:start]+"PALETTE={'fighter_cyan':'#22d3ee','fighter_coral':'#fb7185'}\nANCHORS={}\nNAMES="+repr(names)+'\n'+s[end:];s=s.replace("OUT=ROOT/'outputs/expanded/identity'","OUT=ROOT/'outputs/expanded/mcgregor/tracked/identity'").replace("for sid in 'ABCD':","for sid in 'PQRS':").replace(" PALETTE['fighter_black']='#fb7185' if sid=='B' else '#22d3ee'\n",'').replace('outputs/experiment/body_data/','outputs/expanded/mcgregor/tracked/body_data/').replace('No identity claims for C/D: separate fighter anchors unavailable in initial merged boxes.','Fresh clothing anchors every second; unresolved bodies stay gray.').replace('reviewed A/B clothing anchors','fresh clothing anchors').replace('age<=12','age<=6').replace('(0,24,72,120,180,228)','(0,6,12,18,24,30,36,42)');exec(compile(s,'mcgregor_identity.py','exec'))
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(17,18),(15,19),(16,20),(20,21),(16,22)]
for root in [91,112]:
 for f in range(5):
  a=root+1+f*4;EDGES += [(root,a)]+[(j,j+1) for j in range(a,a+3)]
NOTES={'P':[(0,'Changes the base before attacking.'),(1.25,'Kick → retract → step away; contact unclear.'),(3.375,'A second kick, followed later by a punch.'),(6.375,'Committed punch; feet are cropped here.')],'Q':[(0,'Small shuffles precede the entry.'),(1.25,'Long step into punch → stance recovery.'),(2.375,'Withdraws and resets the distance.'),(4.125,'Punching exchange → backward foot placements.')],'R':[(0,'Standing exchange; hand and leg attempts sampled.'),(3,'Foot visibility is limited; hidden joints are candidates.'),(7.25,'Visible feet reposition; travel direction uncertain.')],'S':[(0,'Repositions after the opponent’s kick sequence.'),(2,'Small changes in foot placement and base width.'),(5.25,'Advancing punches close the distance.'),(6.5,'Continues stepping toward the opponent at the fence.')]}
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);checks=[]
for cid,(year,target,other) in CASES.items():
 poses=json.load(open(O/'body_data'/f'{cid}_rtmw_133.json'));ids=json.load(open(O/'identity'/f'{cid}-identities.json'));cap=cv2.VideoCapture(poses['source']['path']);fps=cap.get(5);path=O/'videos'/f'{year}-tracked.mp4';proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x640','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE);i=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  t=i/fps;ix=min(range(len(poses['frames'])),key=lambda j:abs(poses['frames'][j]['time']-t));row=poses['frames'][ix];ident=ids['frames'][ix];im=Image.new('RGB',(960,640),'#10171c');im.paste(Image.fromarray(cv2.cvtColor(fr[:540],cv2.COLOR_BGR2RGB)),(0,0));d=ImageDraw.Draw(im)
  for j,p in enumerate(row['people']):
   match=next((a for a in ident['people'] if a['detection_index']==j),None);col=match['display_color'] if match else '#a3a3a3';k=p['keypoints'];valid=lambda q: q<len(k) and k[q][2]>=.3 and 0<=k[q][0]<960 and 0<=k[q][1]<540
   for a,z in EDGES:
    if valid(a) and valid(z):d.line((*k[a][:2],*k[z][:2]),fill=col,width=2 if a<23 else 1)
   for q in range(len(k)):
    if valid(q):x,y=k[q][:2];rad=2 if q<23 else 1;d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=col)
  note=next(txt for when,txt in reversed(NOTES[cid]) if t>=when);d.text((15,549),f'{year} · {t:.2f}s · Cyan: McGregor clothing · Coral: opponent · Gray: unknown',font=font,fill='#22d3ee');d.text((15,578),note,font=font,fill='white');d.text((15,607),'6 Hz RTMW candidates; uncertain joints remain estimates. No career trend proven.',font=font,fill='#a3b0b7');proc.stdin.write(np.asarray(im).tobytes());i+=1
  if i==round(fps*2):im.save(O/'videos'/f'{year}-preview.jpg')
 proc.stdin.close();assert proc.wait()==0;cap.release();v=cv2.VideoCapture(str(path));ok,_=v.read();v.set(1,i-1);last,_=v.read();v.release();assert ok and last;checks.append({'year':year,'frames':i,'first_last_decode':True,'colored_coverage':ids['counts']});print('RENDERED',year,flush=True)
(O/'validation.json').write_text(json.dumps({'tracking':'RTMW133 at6Hz; nearest sample overlay; clothing anchors and appearance/motion association with abstention; display threshold not calibrated visibility','detector_sha256':hashlib.sha256(detpath.read_bytes()).hexdigest(),'pose_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'clips':checks},indent=2));print('ALL TRACKED VIDEOS COMPLETE',flush=True)
