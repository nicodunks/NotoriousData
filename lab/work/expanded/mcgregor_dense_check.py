import json,cv2,numpy as np,hashlib,subprocess,time,concurrent.futures
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/dense_check';O.mkdir(exist_ok=True)
for folder in ['evidence','audit','labels']:(O/folder).mkdir(exist_ok=True)
select=[('D01','expansion','Y04'),('D02','pilot','X05'),('D03','expansion','Y11'),('D04','pilot','X11')];cases=[]
for cid,folder,baseid in select:
 B=R/'outputs/expanded/mcgregor'/folder;row=next(r for r in json.load(open(B/'manifest.json')) if r['anonymous_id']==baseid);primary=json.load(open(B/'labels'/f'{baseid}.json'));case={'id':cid,'base_folder':folder,'base_interval':baseid,'source':row['source'],'start':row['start'],'sampling_hz':8,'duration':8,'original_sampling_hz':4,'selection':'First chronologically live, high target identity, usable standing window with >=3s feet visibility among each original per-bout pool; source/frame conditions only, not movement outcome.'};cap=cv2.VideoCapture(str(R/f'work/raw/{row["source"]}.mp4'));tiles=[];images=[]
 for j in range(64):
  cap.set(cv2.CAP_PROP_POS_MSEC,(row['start']+j/8)*1000);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,360));cv2.putText(im,f'{cid} t={j/8:.3f}s',(6,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1);tiles.append(im)
  if len(tiles)==8:
   p=O/'evidence'/f'{cid}-{j//8:02d}.jpg';cv2.imwrite(str(p),np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,8,2)]));images.append(str(p));tiles=[]
 case['images']=images;cases.append(case)
(O/'registered_design.json').write_text(json.dumps({'design':'Sampling-density sensitivity challenge: four phase/visibility-selected windows, same native8s interval, fresh8Hz frames and blind no-history independent labeler. No prior labels, predicted career direction, bout date/name or expected counts supplied. Window units nested in four bouts.','cases':cases},indent=2))
def label(case):
 cid=case['id'];anchor={'obgm6JNtyVo':'NAVY shorts, yellow gloves, RED tape; opponent WHITE shorts and BLUE tape','BCOy-PG8EIw':'BLACK shorts with GREEN patches, RED tape; opponent WHITE patterned shorts, BLUE tape','khabib-mcgregor':'GREEN shorts, BLUE tape; opponent BLACK/GOLD shorts, RED tape','6yu2AWK4rxo':'GREEN/OLIVE shorts, BLUE tape; opponent BLACK shorts with red lettering, RED tape'}[case['source']]
 prompt='''Use ONLY supplied anonymous timestamped frames. No tools, web, names, fight memories, expected styles/era or prior labels. Target anchor: '''+anchor+'''. Count only clearly observed committed target attempts, not merely arm extension, feints or glove proximity. Deduplicate one strike across successive frames. Contact remains unknown unless visible. Footwork must follow visible target feet; camera/body-box motion is not bounce. A bounce cycle needs visible repeated up/down placement, not pose jitter. Report lower bounds when actions can fall between8Hz frames. Independently inspect target action-response sequences: describe approach/withdraw/plant/pivot before/after opponent action only when directly visible; temporal order does not prove causation. Return JSON {"id":"D01","target_identity":"high|medium|low|unresolved","live_fight":"yes|no|unclear","replay":"yes|no|unclear","standing_span":[{"start":0,"end":8}],"feet_visible_seconds_estimate":0,"target_attempts":[{"start":0,"end":1,"type":"punch|kick|knee|elbow|spinning|unclear","certainty":"high|medium|low","evidence":"..."}],"footwork":[{"start":0,"end":1,"pattern":"step|shuffle|bounce_cycle|pivot|plant|withdraw|circle|unclear","count_lower_bound":0,"certainty":"high|medium|low","evidence":"..."}],"action_response":[{"start":0,"end":1,"opponent_observation":"...","target_before":"...","target_after":"...","certainty":"high|medium|low","ambiguity":"..."}],"uncertainties":[]}. All times relative to anonymous interval. No measured3D stance, forces or bodyweight claims.'''
 dest=O/'labels'/f'{cid}.json';log=O/'audit'/f'{cid}.jsonl';(O/'audit'/f'{cid}.prompt.txt').write_text(prompt);cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(R/'work/expanded/isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
 for image in case['images']:cmd+=['-i',image]
 cmd+=['-'];start=time.time()
 with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 ev=[]
 for line in log.read_text().splitlines():
  try:ev.append(json.loads(line))
  except:pass
 used=[e for e in ev if e.get('item',{}).get('type') not in [None,'agent_message','reasoning']];assert r.returncode==0 and not used
 s=dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();dest.write_text(json.dumps(json.loads(s),indent=2));(O/'audit'/f'{cid}.metadata.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','tools_used':used,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'images':[{'path':str(Path(p).relative_to(R)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in case['images']],'seconds':time.time()-start},indent=2));print('DENSE',cid,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(label,cases))
print('DONE',flush=True)
