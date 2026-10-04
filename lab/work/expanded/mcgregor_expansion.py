"""Frozen systematic four-bout expansion; two added bout contexts. No training, no career-trend inference."""
import json,cv2,numpy as np,hashlib,subprocess,time,concurrent.futures,random
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/expansion';O.mkdir(parents=True,exist_ok=True)
for folder in ['evidence','audit','labels','native']:(O/folder).mkdir(exist_ok=True)
SOURCES=[('obgm6JNtyVo',[250,292,334,376,418,460]),('khabib-mcgregor',[625,650,675,700,725,750])]
rows=[]
for sid,starts in SOURCES:
 for st in starts:rows.append({'source':sid,'start':st,'duration':8,'sampling_hz':4})
random.Random(773).shuffle(rows)
for i,row in enumerate(rows):row['anonymous_id']=f'Y{i+1:02d}'
if not (O/'manifest.json').exists():
 freeze={'design':'Six systematically spaced source-time windows per broadcast; same 8s/4Hz extraction. All windows retained, including exclusions. No outcome or movement selection. Anonymous randomized IDs, full native pixels retained; no masking; incomplete blinding because appearances/clothing remain.','sources':SOURCES,'units':'2 added bouts; 12 nested windows, not 384 independent frames','metrics':{'strike_attempts':'visible committed attempts only; feints separate; counts are sampled evidence lower bounds, not full rates','bounce_cycles':'visible repeated up/down footwork cycle only; excludes pose jitter, camera movement; otherwise null','stance_proxy':'2D ankle separation / mean hip-to-ankle distance at visible independently owned paired landmarks; requires full body, quiet stance, adequate projection','guard_proxy':'wrist y minus shoulder midpoint y / torso length; only visible owned wrists; positive means below shoulders; camera-dependent','heel_visibility':'directly seen raised/grounded heel only; no weight or force inference'},'plan_time_unix':time.time()}
 (O/'frozen_sampling.json').write_text(json.dumps({'protocol':freeze,'windows':rows},indent=2))
 for row in rows:
  sid=row['source'];cid=row['anonymous_id'];cap=cv2.VideoCapture(str(R/f'work/raw/{sid}.mp4'));tiles=[];sheets=[]
  for j in range(32):
   t=row['start']+j/4;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read();assert ok
   if j in (0,16,31):cv2.imwrite(str(O/'native'/f'{cid}-{j:02d}.jpg'),im)
   # Blinded annotator input: retain all pixels, including feet; incomplete blinding recorded.
   tile=cv2.resize(im,(640,360));cv2.putText(tile,f'{cid} t={j/4:.2f}s',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1);tiles.append(tile)
   if len(tiles)==8:
    path=O/'evidence'/f'{cid}-{j//8:02d}.jpg';cv2.imwrite(str(path),np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,8,2)]));sheets.append(str(path));tiles=[]
  row['sheets']=sheets;row['native_anchor_frames']=[0,16,31];cap.release()
 (O/'manifest.json').write_text(json.dumps(rows,indent=2))
else:
 rows=json.load(open(O/'manifest.json'))
SCHEMA='''Return JSON {"anonymous_id":"X01","target_clothing":"...","target_identity_visibility":"high|medium|low|absent","live_fight":"yes|no|unclear","likely_replay":"yes|no|unclear","phases":[{"start":0,"end":8,"phase":"standing|clinch|ground|break|nonfight|unclear"}],"camera":{"view":"wide_full_body|medium|close|mixed","cuts":[],"feet_observable_seconds_estimate":0,"camera_motion_confound":"low|medium|high"},"usable_quiet_standing_intervals":[],"strike_attempts":[{"start":0,"end":1,"type":"punch|kick|knee|elbow|spinning|unclear","certainty":"high|medium|low","visible_evidence":"..."}],"feints":[],"footwork":[{"start":0,"end":1,"pattern":"step|shuffle|bounce_cycle|pivot|plant|withdraw|circle|unclear","count_lower_bound":0,"certainty":"high|medium|low","evidence":"..."}],"guard_observations":[{"start":0,"end":1,"observation":"...","certainty":"high|medium|low"}],"stance_observations":[{"start":0,"end":1,"observation":"...","certainty":"high|medium|low"}],"point_anchors":[{"time":0,"eligible_full_body_quiet_stance":false,"reason":"...","points_normalized":{"left_ankle":null,"right_ankle":null,"left_hip":null,"right_hip":null,"left_shoulder":null,"right_shoulder":null,"left_wrist":null,"right_wrist":null},"point_coordinate_uncertainty_normalized":0.02,"visible_owned_points":[]}],"uncertainties":[]}. Point anchors at up to THREE clearly quiet full-body standing timestamps per interval. Choose displayed timestamps INSIDE reported quiet intervals, preferring quiet interval midpoints rather than attack onsets. If no quiet fully visible stance exists, return an empty point_anchors array and explain. Coordinate [x,y] within image tile normalized x/640,y/360; use null for cropped/occluded/uncertain ownership. approximate coordinates are exploratory human-like image estimates, not measured 3D. Quiet stance eligibility false during attacks, kicks, clinch, ground, leg crossing, cropped feet or severe fore/aft foreshortening.'''
PROMPT='''Annotate ONLY supplied timestamped anonymous broadcast frames. No tools, web, files, name identification, known bout memories or expected style hypotheses. Do not infer hidden movements or outcomes. TARGET ANCHOR IS PROVIDED SEPARATELY FOR THIS INTERVAL. Use only that clothing/tape combination, corroborated independently from pixels. Never use black shorts or tape alone as a universal target identifier. Explicitly report actual target clothing and wrist tape; if target is unobservable, abstain. If ambiguous, say unknown. No pixel masking was applied: names and recognizable appearances may reveal identity; do not use them or name identification. Validate the supplied interval-specific shorts and wrist-tape combination separately before target measurements. Observe committed strike attempts, footwork cycles, guard, stance only from pixels. Distinguish camera movement from fighter movement. 4Hz sampling can miss rapid actions/contact; use lower bounds and uncertainty, never claim all strikes counted. Do not assign a named style or infer bodyweight distribution. When no live standing sequence exists, return exclusions rather than manufacture comparison. Each frame shows relative time in its window. '''
def label(row):
 cid=row['anonymous_id'];dest=O/'labels'/f'{cid}.json'
 anchor='NAVY/DARK BLUE shorts, yellow gloves with RED wrist tape, light hair/beard and spine tattoo; opponent wears WHITE shorts and yellow gloves with BLUE tape.' if row['source']=='obgm6JNtyVo' else 'GREEN/OLIVE shorts, BLUE wrist tape, brown hair/beard and chest tattoo; opponent wears BLACK shorts with gold lettering and RED tape.'
 prompt=PROMPT+'\nTarget clothing anchor: '+anchor+'\nAnonymous interval '+cid+'\n'+SCHEMA
 ph=hashlib.sha256(prompt.encode()).hexdigest();imagehash=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in row['sheets']]
 meta_path=O/'audit'/f'{cid}.metadata.json'
 if dest.exists() and meta_path.exists():
  meta=json.load(open(meta_path))
  if meta['prompt_sha256']==ph and [q['sha256'] for q in meta['images']]==imagehash and not meta['tools_used']:return
 # Preserve stale results, prompts and failed logs; never reuse merely by filename.
 superseded=O/'superseded_final_prompt';superseded.mkdir(exist_ok=True)
 import shutil
 for old in [dest,*[O/'audit'/f'{cid}.{suffix}' for suffix in ['prompt.txt','metadata.json','jsonl']]]:
  if old.exists():shutil.move(str(old),str(superseded/(str(time.time_ns())+'-'+old.name)))
 (O/'audit'/f'{cid}.prompt.txt').write_text(prompt);log=O/'audit'/f'{cid}.jsonl';cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(R/'work/expanded/isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
 for path in row['sheets']:cmd+=['-i',path]
 cmd+=['-'];start=time.time()
 with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 ev=[]
 for line in log.read_text().splitlines():
  try:ev.append(json.loads(line))
  except:pass
 used=[e for e in ev if e.get('item',{}).get('type') not in [None,'agent_message','reasoning']]
 assert r.returncode==0 and not used,(cid,r.returncode)
 s=dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();dest.write_text(json.dumps(json.loads(s),indent=2));(O/'audit'/f'{cid}.metadata.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','tools_used':used,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'images':[{'path':str(Path(p).relative_to(R)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in row['sheets']],'seconds':time.time()-start},indent=2));print('LABELED',cid,flush=True)
def safe_label(row):
 for attempt in range(3):
  try:label(row);return
  except Exception as e:
   cid=row['anonymous_id'];(O/'audit'/f'{cid}.failed-{attempt}.txt').write_text(str(e));print('FAILED',cid,attempt,type(e).__name__,flush=True)
   if attempt<2:time.sleep(3)
 print('UNRESOLVED',row['anonymous_id'],flush=True)
if __name__=='__main__':
 runorder=rows
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(safe_label,runorder))
 print('RUN FINISHED',flush=True)

