"""Conditional quiet interval supplement, not representative career sampling."""
import os
import json,cv2,numpy as np,subprocess,hashlib,time,concurrent.futures,torch,onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO
from rtmlib import RTMPose
R=Path.cwd();P=R/'outputs/expanded/mcgregor/expansion';O=R/'outputs/expanded/mcgregor/adaptive_expansion';O.mkdir(exist_ok=True)
for f in ['evidence','audit','poses','labels']:(O/f).mkdir(exist_ok=True)
D={r['anonymous_id']:r for r in json.load(open(P/'manifest.json'))};cases=[]
for cid in ['Y04','Y09','Y11']:
 lab=json.load(open(P/'labels'/f'{cid}.json'));intervals=lab['usable_quiet_standing_intervals'];times=[round(((q['start']+q['end'])/2)*4)/4 for q in intervals];times=[t for t in times if any(q['start']<=t<q['end'] for q in intervals)][:2];cases.append({'id':'Q'+cid[1:],'base_interval':cid,**D[cid],'quiet_times':times,'quiet_intervals':intervals})
(O/'registered_supplement.json').write_text(json.dumps({'design':'Conditional quiet midpoint anchors from already reviewed standing intervals. Original systematic source-time sample and 0/4/7.75 controls preserved. Choose first up to2 quiet intervals per selected standing window, midpoint rounded to nearest4Hz sample; no selection on pose metric value.','cases':cases,'limits':'Conditions on clear quiet visibility; cannot estimate population/career rate or compare named styles.'},indent=2))
cp=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'));torch.set_num_threads(4);det=YOLO(str(R/'work/models/mma-detector/best.pt'));m=RTMPose(str(cp),model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;m.session=ort.InferenceSession(str(cp),sess_options=opts,providers=['CPUExecutionProvider'])
LM={0:'N',5:'LS',6:'RS',9:'LW',10:'RW',11:'LH',12:'RH',15:'LA',16:'RA'}
for case in cases:
 cap=cv2.VideoCapture(case['source_path'] if 'source_path' in case else str(R/f'work/raw/{case["source"]}.mp4'));frames=[];images=[]
 for j,t in enumerate(case['quiet_times']):
  cap.set(cv2.CAP_PROP_POS_MSEC,(case['start']+t)*1000);ok,im=cap.read();assert ok;im=cv2.resize(im,(960,540));native=O/'evidence'/f'{case["id"]}-{j}-native.jpg';cv2.imwrite(str(native),im);overlay=im.copy();rr=det.predict(im,device='mps',conf=.4,imgsz=960,verbose=False)[0];boxes=rr.boxes.xyxy.cpu().numpy();xy,sc=m(im,boxes.tolist()) if len(boxes) else ([],[]);ps=[]
  for index,(b,k,s) in enumerate(zip(boxes,xy,sc)):
   ps.append({'detection_index':index,'box':b.round(2).tolist(),'keypoints':np.c_[k,s].round(3).tolist()});col=(238,211,34) if index%2==0 else (133,113,251);cv2.rectangle(overlay,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),col,2);cv2.putText(overlay,f'D{index}',tuple((b[:2]+[0,20]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,col,2)
   for q,label in LM.items():
    cv2.circle(overlay,tuple(k[q].astype(int)),3,col,-1);cv2.putText(overlay,f'{index}:{label}',tuple((k[q]+[3,-3]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)
  cv2.putText(overlay,f'{case["id"]} t={t:.2f}s: raw unverified candidates',(5,525),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);path=O/'evidence'/f'{case["id"]}-{j}-pose.jpg';cv2.imwrite(str(path),overlay);images += [str(native),str(path)];frames.append({'local_time':t,'source_time':case['start']+t,'people':ps})
 case['images']=images;case['frames']=frames;(O/'poses'/f'{case["id"]}.json').write_text(json.dumps({'width':960,'height':540,'rtmw_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'frames':frames,'raw_scores':'SIMCC peaks; not calibrated probabilities'},separators=(',',':')))

def review(case):
 cid=case['id'];anchor='NAVY/DARK BLUE shorts, yellow gloves with RED wrist tape; opponent WHITE shorts with BLUE tape' if case['source']=='obgm6JNtyVo' else 'GREEN/OLIVE shorts, BLUE wrist tape; opponent BLACK shorts with gold lettering and RED tape';prompt='''Use ONLY supplied original and labeled candidate-pose images. No tools, names, web, fight memories, date/era or expected style. Target clothing anchor: '''+anchor+'''. Native+pose images alternate in time order. Review each timestamp separately for visible ownership and anatomical plausibility. Candidate label keys N nose, LS/RS shoulders, LW/RW wrists, LH/RH hips, LA/RA ankles. Detector number is not identity. For each landmark, accept only if visibly belongs to target and is plausibly on intended anatomical region; unobservable or mixed ownership is false. Approximate hip centers cannot establish actual skeletal joints. Quiet intervals were selected independently of metric values. Return JSON {"frames":[{"local_time":0,"target_detection_index":null,"target_identity":"high|medium|low|unresolved","standing":true,"quiet":true,"feet_visible":true,"view":"front|back|side|three_quarter|unclear","foreshortening":"low|medium|high","accepted_landmarks":{"N":false,"LS":false,"RS":false,"LW":false,"RW":false,"LH":false,"RH":false,"LA":false,"RA":false},"heel_contact":"both_flat|one_raised|both_raised|unclear","evidence":"specific visible cues; failures","uncertainties":[]}],"limits":[]}. Do not force completeness or convert pose confidence to visibility.'''
 prompt+='\nTimestamp list: '+json.dumps(case['quiet_times']);dest=O/'labels'/f'{cid}.json';log=O/'audit'/f'{cid}.jsonl';(O/'audit'/f'{cid}.prompt.txt').write_text(prompt);cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(R/'work/expanded/isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
 for im in case['images']:cmd+=['-i',im]
 cmd+=['-'];start=time.time()
 with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 ev=[]
 for line in log.read_text().splitlines():
  try:ev.append(json.loads(line))
  except:pass
 used=[e for e in ev if e.get('item',{}).get('type') not in [None,'agent_message','reasoning']];assert r.returncode==0 and not used
 s=dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();dest.write_text(json.dumps(json.loads(s),indent=2));(O/'audit'/f'{cid}.metadata.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','tools_used':used,'seconds':time.time()-start,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'images':[{'path':str(Path(p).relative_to(R)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in case['images']]},indent=2));print('REVIEWED',cid,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(review,cases))
(O/'manifest.json').write_text(json.dumps(cases,indent=2));print('DONE',flush=True)
