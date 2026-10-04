import json,cv2,subprocess,hashlib,time,concurrent.futures
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path.cwd(); O=R/'outputs/expanded'; W=R/'work/expanded';
for p in [O/'scouting',O/'scout_evidence',O/'scout_audit',W/'isolated']:p.mkdir(parents=True,exist_ok=True)
S=[('pereira-adesanya',[80,180,340,500,660,820,980,1140]),('khabib-mcgregor',[80,180,280,380,500,620,760,900]),('oliveira-poirier',[80,160,240,320,480,560,620,720]),('holloway-gaethje-highlights',[15,55,95,135,175,215])]
manifest=[]
for src,starts in S:
 p=R/'work/raw'/f'{src}.mp4'; cap=cv2.VideoCapture(str(p)); fps=cap.get(5); dur=cap.get(7)/fps
 for st in starts:
  cid=f'S{len(manifest)+1:02d}'; end=min(st+24,dur); frames=[]
  im=Image.new('RGB',(1920,1220),'#101712');d=ImageDraw.Draw(im)
  for n in range(9):
   t=st+n*(end-st)/9; cap.set(0 if False else cv2.CAP_PROP_POS_MSEC,t*1000);ok,f=cap.read()
   if not ok:raise RuntimeError((src,t))
   x=n%3*640;y=n//3*400; im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f,(640,360)),cv2.COLOR_BGR2RGB)),(x,y+30));d.text((x+10,y+7),f'clip {cid}: local {t-st:.3f}s',fill='white');frames.append(dict(local_time=t-st,source_time=t))
  path=O/'scout_evidence'/f'{cid}.jpg'; im.save(path,quality=94)
  manifest.append(dict(id=cid,source=src,source_path=str(p),start=st,end=end,duration=end-st,fps=fps,width=int(cap.get(3)),height=int(cap.get(4)),sample_frames=frames,evidence=str(path.relative_to(O)),evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sampling='Systematic source-time windows; not selected for dramatic action. Sparse scouting only.'))
 cap.release()
(O/'scout_manifest.json').write_text(json.dumps(manifest,indent=2))
PROMPT='''Use ONLY the attached ordered MMA frames. No tools, names, known fights, web, memories, or future inference. This is SPARSE SCOUTING, NOT precise strike/contact annotation. Keep full-body/clothing identity, not left/right. If replay/camera cuts/nonfight appear, note them. Return valid JSON with these fields: clip_id, live_fight (yes/no/unclear), likely_replay (yes/no/unclear), camera_cut_between_samples (boolean), fighters (array of objects: color_id, clothing, stable_identity_evidence), phases_seen (array of standing/clinch/ground/unclear), visible_sequence (array of time_start,time_end,actor,color_action_description), useful_for (array of movement/strike_contact/clinch/ground_control/submission/fighter_evolution/prediction/occlusion), contact_scouting_limit (string), uncertainties (array). Do not assign precise impact, damage, or clean/block from sparse evidence. Mark potential moments only.'''
def run(row):
 tag=row['id'];out=O/'scouting'/f'{tag}.json';prompt=PROMPT+'\nclip_id='+tag+'; duration='+str(row['duration'])+'; each frame local timestamp is shown.'; ph=hashlib.sha256(prompt.encode()).hexdigest()
 if out.exists():
  cached=json.loads((O/'scout_audit'/f'{tag}.metadata.json').read_text())
  assert cached['prompt_sha256']==ph and cached['evidence_sha256']==row['evidence_sha256'] and cached['model']=='gpt-6.1-sol' and cached['reasoning']=='high' and not cached['tools_used'], 'Cached scout provenance mismatch'
  return tag
 log=O/'scout_audit'/f'{tag}.jsonl';(O/'scout_audit'/f'{tag}.prompt.txt').write_text(prompt)
 cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(W/'isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(out),'-i',str(O/row['evidence']),'-']; t=time.time()
 for attempt in range(3):
  with log.open('w') as f:r=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=420)
  if r.returncode==0 and out.exists():
   try:
    obj=json.loads(out.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip());assert obj['clip_id']==tag
    lines=[]
    for l in log.read_text().splitlines():
     try:lines.append(json.loads(l))
     except ValueError:pass
    tu=[x for x in lines if x.get('item',{}).get('type') not in (None,'reasoning','agent_message')]; assert not tu
    out.write_text(json.dumps(obj,indent=2));(O/'scout_audit'/f'{tag}.metadata.json').write_text(json.dumps(dict(model='gpt-6.1-sol',reasoning='high',prompt_sha256=ph,evidence_sha256=row['evidence_sha256'],seconds=time.time()-t,tools_used=tu),indent=2));print('SCOUT',tag,obj['live_fight'],obj['phases_seen'],flush=True);return tag
   except Exception as e: print('RETRY',tag,str(e),flush=True);out.unlink(missing_ok=True)
 raise RuntimeError(tag)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,manifest))
 print('SCOUT COMPLETE',len(manifest),flush=True)
