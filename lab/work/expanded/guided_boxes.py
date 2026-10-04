import cv2,json,subprocess,hashlib,time,concurrent.futures
from pathlib import Path
ROOT=Path.cwd();OUT=ROOT/'outputs/expanded/identity';AUDIT=OUT/'guided_audit';AUDIT.mkdir(exist_ok=True)

def run(sid):
 d=json.load(open(ROOT/f'outputs/experiment/body_data/{sid}_rtmw_133.json'));cap=cv2.VideoCapture(d['source']['path']);tiles=[]
 for frame in range(0,240,48):
  row=d['frames'][frame];cap.set(cv2.CAP_PROP_POS_MSEC,row['time']*1000);ok,im=cap.read()
  for j,p in enumerate(row['people']):
   x1,y1,x2,y2=map(int,p['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(255,255,0),2);cv2.putText(im,f'D{j}',(x1,y1+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
  cv2.putText(im,f'{sid} frame {frame} t={row["time"]:.2f}s',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,255),2);tiles.append(cv2.resize(im,(960,540)))
 sheet=AUDIT/f'{sid}-numbered.jpg';cv2.imwrite(str(sheet),__import__('numpy').vstack([__import__('numpy').hstack(tiles[i:i+2]) for i in range(0,4,2)]))
 identities={'A':['fighter_green','fighter_yellow'],'B':['fighter_green','fighter_black'],'C':['fighter_black','fighter_red'],'D':['fighter_black','fighter_red']}[sid]
 prompt='Use ONLY this supplied numbered native-resolution image sheet (each tile 960x540; displayed frame timestamps). No tools, web, files or fight memories. Return valid JSON {"frames":[{"frame":0,"fighters":[{"identity":"fighter_black|fighter_red","box_normalized":[0.1,0.2,0.6,0.8],"certainty":"high|medium|low","visible_evidence":"clothing/hair and body cues","occluded_boundaries":[],"pose_ownership_risk":"low|medium|high"}]}],"limitations":[]}. For EACH displayed frame, estimate separate anatomical bounding boxes for BOTH fighters from visible body extents and clothing. Coordinates within its tile normalized x/960,y/540. Black/gray shorts and light hair belongs fighter_black; red shorts belongs fighter_red. Do not confuse referee with fighter. Include occluded or overlapping extents explicitly as uncertain, do not claim visible hidden hands/feet. If no separate body can be estimated reliably, box_normalized:null. Existing numbered detector boxes often merge bodies; your task is independent separate fighter box guidance. All four displayed frames required.'

 pf=AUDIT/f'{sid}.prompt.txt';pf.write_text(prompt);dest=AUDIT/f'{sid}.json';log=AUDIT/f'{sid}.jsonl';start=time.time()
 cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(ROOT/'work/experiment/isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest),'-i',str(sheet),'-']
 with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=420)
 events=[]
 for line in log.read_text().splitlines():
  try:events.append(json.loads(line))
  except:pass
 used=[x for x in events if x.get('item',{}).get('type') not in [None,'agent_message','reasoning']]
 meta={'model':'gpt-6.1-sol','reasoning':'high','tools_used':used,'seconds':time.time()-start,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'image_sha256':hashlib.sha256(sheet.read_bytes()).hexdigest(),'returncode':r.returncode}
 (AUDIT/f'{sid}.metadata.json').write_text(json.dumps(meta,indent=2));assert not used and r.returncode==0
 text=dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();dest.write_text(json.dumps(json.loads(text),indent=2));print('ANCHOR',sid,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,'CD'))
