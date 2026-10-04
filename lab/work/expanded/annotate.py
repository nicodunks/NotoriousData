"""Expanded evidence annotation: raw resolution, discovery, native-rate ambiguity review; no training."""
import cv2,json,hashlib,subprocess,time,concurrent.futures,sys,math,os,datetime
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg
R=Path.cwd(); W=R/'work/expanded'; O=R/'outputs/expanded'
for d in ['clips','evidence','annotations','audit']:(O/d).mkdir(parents=True,exist_ok=True)
(W/'isolated').mkdir(exist_ok=True)
S=[('E','pereira-adesanya',100,'https://www.youtube.com/watch?v=9gLe0JfKz8w'),('F','pereira-adesanya',500,'https://www.youtube.com/watch?v=9gLe0JfKz8w'),('G','khabib-mcgregor',200,'https://www.youtube.com/watch?v=JuBBIJ7adjM'),('H','oliveira-poirier',550,'https://www.youtube.com/watch?v=clG3LV28bC0'),('I','GoUjizSdLEw',486,'https://www.youtube.com/watch?v=GoUjizSdLEw'),('J','uJBmkeUVM6s',397,'https://www.youtube.com/watch?v=uJBmkeUVM6s')]
RULES='''Annotate anonymous MMA evidence, using only supplied ordered timestamped images. No tools, files, web, fighter names, fight memory or future result. Return JSON only. Use explicit clothing-color names as primary identity keys (e.g. Green/Yellow, Black/White), never screen side; link each clothing ID to stable Cyan/Coral figure display palette. assign the two figures Cyan #22d3ee and Coral #fb7185. These are display IDs, not actual clothing colors. Say identity unclear when occluded. Contact labels ONLY evidence_supported_contact, evidence_supported_miss, unclear, not_applicable. Contact does not establish clean/block/damage. Distinguish attempted movement from completed outcome. Include visible face/hands/feet and their visibility/ownership limits. No training and no gold-truth claim.'''
def call(tag,prompt,images):
 dest=O/'audit'/f'{tag}.json';pf=O/'audit'/f'{tag}.prompt.txt';log=O/'audit'/f'{tag}.jsonl';meta=O/'audit'/f'{tag}.metadata.json'
 if dest.exists() and meta.exists():
  m=json.loads(meta.read_text());assert m['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest();assert m['model']=='gpt-6.1-sol' and m['reasoning']=='high' and not m['tools_used'];assert [i['sha256'] for i in m['images']]==[hashlib.sha256(i.read_bytes()).hexdigest() for i in images],'Cached image mismatch; use a fresh output path for rerun';return json.loads(dest.read_text())
 pf.write_text(prompt);start=time.time()
 cmd=['caffeinate','-i','codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(W/'isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(dest)]
 for im in images:cmd+=['-i',str(im)]
 cmd+=['-']
 for attempt in range(3):
  deadline=os.environ.get('EXPT_DEADLINE_UTC')
  remaining=(datetime.datetime.fromisoformat(deadline).timestamp()-time.time()) if deadline else 1800
  if remaining<=0:raise RuntimeError('User requested experiment deadline reached; preserve incomplete status')
  try:
   with log.open('w') as f:res=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=min(1800,max(1,remaining)))
   if res.returncode==0 and dest.exists():break
  except subprocess.TimeoutExpired:res=None
  log.replace(O/'audit'/f'{tag}.failed_attempt_{int(time.time())}_{attempt}.jsonl')
 else:raise RuntimeError(f'{tag}: three transport/execution failures; no annotation asserted')
 obj=json.loads(dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip());lines=[]
 for l in log.read_text().splitlines():
  try:lines.append(json.loads(l))
  except:pass
 used=[x for x in lines if x.get('item',{}).get('type') not in [None,'reasoning','agent_message','error']]
 if used:raise RuntimeError('Tools used')
 dest.write_text(json.dumps(obj,indent=2));meta.write_text(json.dumps(dict(model='gpt-6.1-sol',reasoning='high',seconds=time.time()-start,tools_used=used,transport_errors=[x for x in lines if x.get('item',{}).get('type')=='error'],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),images=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],usage=[x.get('usage') for x in lines if x.get('usage')]),indent=2));return obj

def prepare(sources=None,manifest_name="annotation_manifest.json"):
 sources=S[:4] if sources is None else sources
 manifest=[]
 for cid,src,offset,url in sources:
  cap=cv2.VideoCapture(str(R/'work/raw'/f'{src}.mp4'));fps=cap.get(5);width=int(cap.get(3));height=int(cap.get(4));rowheight=height+28;ew=O/'evidence'/cid;ew.mkdir(exist_ok=True);sheets=[]
  for st in range(0,20,4):
   sheet=Image.new('RGB',(width*2,4*rowheight),'#111');dr=ImageDraw.Draw(sheet)
   for j in range(8):
    t=st+j/2;cap.set(0 if False else cv2.CAP_PROP_POS_MSEC,(offset+t)*1000);ok,f=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));x=j%2*width;y=j//2*rowheight;sheet.paste(im,(x,y));dr.text((x+12,y+height+5),f'{cid} relative {t:.3f}s source {offset+t:.3f}s',fill='white')
   p=ew/f'discovery_{st:02d}.jpg';sheet.save(p,quality=96);sheets.append(str(p.relative_to(R)))
  cap.release();clip=O/'clips'/f'{cid}.mp4'
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss',str(offset),'-i',str(R/'work/raw'/f'{src}.mp4'),'-t','20','-an','-c:v','libx264','-crf','18',str(clip)],check=True)
  manifest.append(dict(id=cid,source=src,source_url=url,source_offset=offset,duration=20,source_fps=fps,width=width,height=height,discovery_hz=2,discovery_crop='full native frames; no masking or spatial downscale',clip=str(clip.relative_to(R)),discovery_sheets=sheets,selection='Preselected broadly distributed ordinary offsets, distinct from A-D; live-fight eligibility determined from discovery images'))
 (O/manifest_name).write_text(json.dumps(manifest,indent=2))
def annotate(cid):
 images=sorted((O/'evidence'/cid).glob('model_discovery*.jpg')) or sorted((O/'evidence'/cid).glob('discovery*.jpg'))
 schema='''Return {clip_id,live_fight:{eligible:boolean,evidence_timestamps:[],reason},fighters:[{id:clothing_color_name,clothing,figure_color:Cyan_or_Coral,display_hex}],visibility:{face,hands,feet,occlusions},segments:[{start,end,phase,position,control,movement,action_response,outcome,evidence_timestamps,uncertainties}],events:[{id,start,end,actor,action,target,response,contact,position_before,position_after,control_change,visible_outcome,evidence_timestamps,certainty,uncertainties}],dense_requests:[{start,end,reason}],limits:[]}. Describe all 20 seconds, including quiet periods and camera cuts. Every substantive observation needs relative timestamps. Request at most three short 0.6-second native-rate windows around ambiguous attempted contact/control, bounded [0,20].'''
 primary=call(f'{cid}_discovery',RULES+'\n'+schema,images);print(cid,'discovery',flush=True)
 # Independent reviewer receives frames and schema only, never primary labels.
 independent=call(f'{cid}_independent',RULES+'\nFresh independent annotation; '+schema,images);print(cid,'independent',flush=True)
 windows=[]
 for req in (primary.get('dense_requests',[])+independent.get('dense_requests',[])):
  st=max(0,min(19.4,float(req.get('start',0))));en=min(20,st+.6)
  if all(abs(st-a)>0.4 for a,b in windows):windows.append((st,en))
  if len(windows)==4:break
 src=next(s for s in S if s[0]==cid);cap=cv2.VideoCapture(str(R/'work/raw'/f'{src[1]}.mp4'));fps=cap.get(5);dense=[];dm=[]
 for k,(st,en) in enumerate(windows):
  first=math.floor(st*fps);last=math.ceil(en*fps);frames=[]
  for idx in range(first,last):
   t=idx/fps;cap.set(cv2.CAP_PROP_POS_MSEC,(src[2]+t)*1000);ok,f=cap.read()
   if ok:frames.append((t,f))
  # Preserve full native width and face/hands/feet; each page has four frames, avoids server image shrink of huge sheets.
  for page in range(0,len(frames),4):
   group=frames[page:page+4];height,width=group[0][1].shape[:2];rowheight=height+28;im=Image.new('RGB',(width*2,2*rowheight),'#111');dr=ImageDraw.Draw(im)
   for j,(t,f) in enumerate(group):
    x=j%2*width;y=j//2*rowheight;im.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(x,y));dr.text((x+12,y+height+5),f'{cid} native frame {round((src[2]+t)*fps)} relative {t:.4f}s',fill='white')
   p=O/'evidence'/cid/f'dense_{k}_{page//4:02d}.jpg';im.save(p,quality=97);dense.append(p)
  dm.append(dict(start=st,end=en,source_start=src[2]+st,fps=fps,frame_count=len(frames),crop='full native frames; no masking'))
 cap.release();(O/'evidence'/cid/'dense_manifest.json').write_text(json.dumps(dm,indent=2))
 final=call(f'{cid}_adjudication',RULES+'\nCompare the two independent discovery annotations against supplied native-rate evidence. Neither annotation is authoritative. Preserve disagreements and uncertainty; no majority-vote ground truth. Return {clip_id,fighters,live_fight,segments,events,disagreements:[{claim,primary,independent,native_evidence_timestamps,resolution,remaining_uncertainty}],visibility,limits}. Include evidence timestamps and contact taxonomy. The dense windows cover only part of clip, retain unverified discovery observations with sampling limitations. PRIMARY '+json.dumps(primary)+'\nINDEPENDENT '+json.dumps(independent),dense or images)
 (O/'annotations'/f'{cid}.json').write_text(json.dumps(final,indent=2));print(cid,'COMPLETE',len(dense),'dense sheets',flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(annotate,'EFGH'))
