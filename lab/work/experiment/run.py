import json, subprocess, time, hashlib, concurrent.futures, threading
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/'work/experiment';OUT=ROOT/'outputs/experiment'; LOCK=threading.Lock()
for d in ['labels','verification','predictions','audit']:(OUT/d).mkdir(exist_ok=True)
RULES='''You are annotating an anonymous MMA clip scientifically. Use ONLY the supplied ordered frames and prior annotations. DO NOT use tools, web, files, known fight memories, fighter names or infer a future outcome. Images are timestamped 8 Hz; contact between samples may be unobservable. Describe visible evidence, never assume landed from limb proximity. Pose confidence is not truth. Use unknown/unclear liberally. Alpha/Beta identifiers must follow clothing descriptions, never left/right. Return ONLY valid JSON, no markdown.'''
def call(tag,prompt,images=()):
 dest=OUT/'audit'/f'{tag}.json'; log=OUT/'audit'/f'{tag}.jsonl'; promptfile=OUT/'audit'/f'{tag}.prompt.txt';
 if dest.exists():
  meta=json.loads((OUT/'audit'/f'{tag}.metadata.json').read_text())
  assert meta['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest(), 'Cached prompt mismatch'
  assert meta['model']=='gpt-6.1-sol' and meta['reasoning']=='high' and not meta['tools_used']
  assert promptfile.read_text()==prompt, 'Saved prompt differs'
  return json.loads(dest.read_text())
 promptfile.write_text(prompt)
 cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(BASE/'isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
 for im in images:cmd+=['-i',str(im)]
 cmd+=['-']
 start=time.time()
 for attempt in range(3):
  with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=420)
  if r.returncode==0 and dest.exists():
   try:
    txt=dest.read_text().strip();txt=txt.removeprefix('```json').removeprefix('```').removesuffix('```').strip();obj=json.loads(txt)
    lines=[]
    for line in log.read_text().splitlines():
     try:lines.append(json.loads(line))
     except ValueError:pass
    tools=[x for x in lines if x.get('item',{}).get('type') not in [None,'agent_message','reasoning']]
    if tools:raise ValueError('Tool use detected: '+str(tools)[:200])
    dest.write_text(json.dumps(obj,indent=2));(OUT/'audit'/f'{tag}.metadata.json').write_text(json.dumps(dict(model='gpt-6.1-sol',reasoning='high',seconds=time.time()-start,images=[str(i.relative_to(ROOT)) for i in images],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),tools_used=tools,usage=[x.get('usage') for x in lines if x.get('usage')]),indent=2));return obj
   except Exception as e:
    if dest.exists():dest.unlink()
    with LOCK:print(tag,'retry',str(e),flush=True)
  else:
   with LOCK:print(tag,'failed',r.returncode,log.read_text()[-700:],flush=True)
 raise RuntimeError(tag)
LABEL_TEMPLATE='''Return this schema exactly (replace values): {"start":0,"end":4,"fighters":{"Alpha":"distinctive clothing","Beta":"distinctive clothing"},"start_phase":"standing|clinch|ground|unclear","end_phase":"standing|clinch|ground|unclear","position":"visible relative position and control, or unclear","movement":"visible approach/retreat/circling/control changes","events":[{"time_start":0.0,"time_end":0.5,"actor":"Alpha|Beta|unclear","action":"specific observable attempted action","target":"head|body|leg|position|unclear","contact":"clean|partial|blocked|missed|unclear|not_applicable","evidence":"timestamps and observable cue","certainty":"high|medium|low"}],"event_flags":{"strike_attempt":"yes|no|unclear","takedown_attempt":"yes|no|unclear","ground_position_change":"yes|no|unclear"},"visibility":"good|partial|poor","uncertainties":["important limits"]}. ground_position_change means meaningful change of ground position/control, not every adjustment. Strike attempt includes kicks/knees/elbows/ground strikes, not feints alone. Takedown attempt requires visible committed entry/throw/trip, not level-change alone. No event is a valid result.'''
def label_clip(cid):
 history=[]
 for st in range(0,20,4):
  images=[BASE/'frames'/cid/f'{j:02d}-{j+2:02d}.jpg' for j in [st,st+2]]
  prompt=RULES+'\nLabel ONLY interval '+str((st,st+4))+'. Future frames are not supplied. Previous causal annotations: '+json.dumps(history)+'\n'+LABEL_TEMPLATE
  obj=call(f'label_{cid}_{st:02d}',prompt,images);obj['start']=st;obj['end']=st+4;history.append(obj)
  (OUT/'labels'/f'{cid}.json').write_text(json.dumps(history,indent=2))
  with LOCK:print('LABEL',cid,st+4,flush=True)
 return cid
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(label_clip,'ABCD'))
 print('CAUSAL LABELS COMPLETE',flush=True)
