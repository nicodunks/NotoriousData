"""Frozen causal sequence experiments on E-H; no training; no raw future in predictors."""
import json,cv2,subprocess,time,hashlib,concurrent.futures,math
from pathlib import Path
from PIL import Image,ImageDraw
R=Path.cwd();O=R/'outputs/expanded/causal_v2';W=R/'work/expanded';O.mkdir(exist_ok=True)
for d in ['evidence','observations','references','predictions','audit']:(O/d).mkdir(exist_ok=True)
PHASES=['standing','clinch','ground','unclear'];ACTIONS=['punch','kick_knee_elbow','takedown_entry','ground_strike','control_change','submission_entry','none','unclear']
MAN=json.loads((R/'outputs/expanded/annotation_manifest.json').read_text());manifest={r['id']:r for r in MAN};pose=json.loads((R/'outputs/expanded/poses/E-H.json').read_text())['clips']
protocol={'model':'gpt-6.1-sol','reasoning':'high','clips':list('EFGH'),'cutoffs':[4,8,12,16],'horizon_seconds':4,'conditions':['coarse_history','rich_history','rich_pose_history'],'training':False,'causal_observation_input':'4Hz full-frame image sequences for each prior4-secondbin; labels sequential and supplied only prior labels. Full future pixels forbidden.','reference_input':'Separate independent model reviews upcoming4-secondwindow, 4Hz fullframes; model-reviewed not experttruth. Unknown first actions unscoreable. Contact uncertainty excluded as targets.','sampling':'Preselected E-H ordinary livefight clips from earlier expansion; selected dependent windows, nested within3 bouts. This is not a population sample.','pose':'12Hz existing RTMW sampling, compact2Hz features within most recent4seconds, all ownership unverified; rawscores notprobabilities.','normalization':'All finite nonnegative categorical distributions must sum to1 within rounding tolerance0.02; renormalizeall equally and preserve rawvalues.','targets':{'end_phase':PHASES,'first_new_committed_action':ACTIONS},'metrics':['top1accuracy','categoricalBrier','pairedpercaseBrierdifferences','phasepersistence','alwaysnone'],'phase_persistence':'Most recentlyobserved phase; onehot prediction, unknown remainsunknown','guardrail':'No future target/text/rawframes/identities/names/date/externalfightknowledge in predictor prompts. References frozen and separatelyhashed. No web/tools duringcall.','created_unix':time.time()}
if not (O/'protocol.json').exists():(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
def call(tag,prompt,images=()):
 p=O/'audit'/f'{tag}.json';m=O/'audit'/f'{tag}.metadata.json';h=hashlib.sha256(prompt.encode()).hexdigest();ih=[{'path':str(q.relative_to(R)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in images]
 if p.exists() and m.exists():
  meta=json.loads(m.read_text());assert meta['prompt_sha256']==h and meta['images']==ih and not meta['tools_used'];return json.loads(p.read_text())
 (O/'audit'/f'{tag}.prompt.txt').write_text(prompt);log=O/'audit'/f'{tag}.jsonl';cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C',str(W/'isolated'),'-s','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(p)]
 for q in images:cmd+=['-i',str(q)]
 cmd+=['-'];t=time.time()
 for attempt in range(3):
  if log.exists():log.rename(log.with_name(log.stem+f'.previous-{time.time_ns()}.jsonl'))
  try:
   with log.open('w') as f:res=subprocess.run(cmd,input=prompt,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  except subprocess.TimeoutExpired:
   print('TIMEOUT',tag,'attempt',attempt+1,flush=True);p.unlink(missing_ok=True);continue
  if res.returncode==0 and p.exists():
   try:
    obj=json.loads(p.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip());lines=[]
    for l in log.read_text().splitlines():
     try:lines.append(json.loads(l))
     except ValueError:pass
    tu=[x for x in lines if x.get('item',{}).get('type') not in [None,'reasoning','agent_message','error']];assert not tu
    p.write_text(json.dumps(obj,indent=2));m.write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','prompt_sha256':h,'images':ih,'tools_used':tu,'transport_errors':[x for x in lines if x.get('item',{}).get('type')=='error'],'seconds':time.time()-t},indent=2));return obj
   except Exception as e:print('RETRY',tag,str(e),flush=True)
  p.unlink(missing_ok=True)
 raise RuntimeError(tag)
def evidence(cid,st):
 cap=cv2.VideoCapture(str(R/manifest[cid]['clip']));width=int(cap.get(3));height=int(cap.get(4));images=[]
 for sec in range(st,st+4):
  out=O/'evidence'/f'{cid}_{sec:02d}.jpg'
  if not out.exists():
   #Four full frames perpage, fixed 960x540 preview each, preserving fullbody/no masking.
   im=Image.new('RGB',(1920,1136),'#10171c');d=ImageDraw.Draw(im)
   for n in range(4):
    t=sec+n/4;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,f=cap.read();assert ok
    tile=Image.fromarray(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB));x=n%2*960;y=n//2*568;im.paste(tile,(x,y));d.text((x+12,y+546),f'{cid} t={t:.3f}s (4Hz fullframe, no mask)',fill='white')
   im.save(out,quality=96)
  images.append(out)
 cap.release();return images
RULE='Use ONLY supplied ordered frames and explicit previous observations. No tools, web, files, known fight memory, names, inference of future outcomes. Clothing color IDs, never Alpha/Beta or fixed screen side. Images sampled4Hz may miss rapid actions. Describe visibility/ownership ambiguity; rawposeconfidence is not truth. Return JSON only.'
LABEL='Return {start,end,fighters:[{color_id,clothing}],end_phase,position,movement,guard,footwork,events:[{start,end,actor,action,target,response,contact:contact|miss|unclear|not_applicable,evidence,certainty}],visibility,uncertainties}. No clean/block/damage inference. Phase standing/clinch/ground/unclear. Only observed facts through interval end, no future speculation.'
REF='Return {start,end,end_phase,first_new_committed_action,first_action_onset,first_action_evidence,first_action_certainty,uncertainties}. phase standing/clinch/ground/unclear. first_new_committed_action oneof '+','.join(ACTIONS)+'. First action means first visiblyNEW onset inside window, excludes ongoing movement already inprogress at firstframe. Guard shifts, feints and small adjustments are not committedaction. control_change means meaningful clinch/ground controltransition or escape, not every grip adjustment. takedown_entry committed shot/throw/trip; submission_entry newly established visible submissionattack. If order/onset ambiguous chooseunclear, not none. none onlywhen enough continuoussampledevidence shows no committedaction; sparse samplinglimits explicit. No winner/outcome memories.'
def label(cid):
 hist=[]
 for st in range(0,20,4):
  a=call(f'{cid}_obs_{st}',RULE+'\nLabelONLY '+str([st,st+4])+'. Previous CAUSAL observations '+json.dumps(hist)+'\n'+LABEL,evidence(cid,st));a['start']=st;a['end']=st+4;hist.append(a);(O/'observations'/f'{cid}.json').write_text(json.dumps(hist,indent=2));print('OBS',cid,st,flush=True)
 return cid
def reference_evidence(cid,t):
 cap=cv2.VideoCapture(str(R/manifest[cid]['clip']));fps=cap.get(5);n=int(cap.get(7));paths=[]
 for tag,times in [('preroll',[t-.5,t-.25]),('endpoint',[min(t+4,(n-1)/fps)])]:
  im=Image.new('RGB',(960*len(times),568),'#10171c');d=ImageDraw.Draw(im)
  for j,when in enumerate(times):
   ix=min(n-1,round(when*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,f=cap.read();assert ok
   im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB)),(j*960,0));d.text((j*960+10,546),f'{cid} reference-only {tag} actual t={ix/fps:.4f}s',fill='white')
  p=O/'evidence'/f'{cid}_ref_{t}_{tag}.jpg';im.save(p,quality=96);paths.append(p)
 cap.release();return [paths[0]]+evidence(cid,t)+[paths[1]]
def reference(case):
 cid,t=case;a=call(f'{cid}_ref_{t}',RULE+'\nIndependent reference review only '+str([t,t+4])+'. No priorannotations supplied. Reference-only pre-roll helps distinguish alreadyactive actions; do not score pre-roll actions. Endpoint image gives last available phase at/before horizon, with actual timestamp. New action exactly at end boundary is outside firstactiontarget.\n'+REF,reference_evidence(cid,t));assert a['end_phase'] in PHASES and a['first_new_committed_action'] in ACTIONS;a['start']=t;a['end']=t+4;(O/'references'/f'{cid}_{t}.json').write_text(json.dumps(a,indent=2));print('REF',cid,t,flush=True)
def pose_prefix(cid,t):
 rows=pose[cid]['frames'];pick=[]
 for when in [t-4+i*.5 for i in range(8)]:
  r=min((r for r in rows if r['capture_time_seconds']<t),key=lambda r:abs(r['capture_time_seconds']-when));people=[]
  for z in r['people'][:4]:
   k=z['keypoints_xy_score'];js=list(range(23))+[91,95,99,103,107,111,112,116,120,124,128,132];people.append({'box':z['box_xyxy_pixels'],'keypoints_indices':js,'keypoints':[k[j] for j in js],'ownership':'unverified detection, not stablefighteridentity'})
  pick.append({'time':r['capture_time_seconds'],'people':people})
 return {'width':pose[cid]['width'],'height':pose[cid]['height'],'raw_scores_not_probabilities':True,'frames':pick}
def predict(case):
 cid,t,cond=case;allpast=json.loads((O/'observations'/f'{cid}.json').read_text());hist=[a for a in allpast if a['end']<=t]
 assert len(hist)==t//4
 if cond=='coarse_history':hist=[{'start':a['start'],'end':a['end'],'end_phase':a['end_phase'],'visibility':a.get('visibility')} for a in hist]
 inp={'past_observations':hist}
 if cond=='rich_pose_history':inp['past_pose']=pose_prefix(cid,t)
 assert all(a['end']<=t for a in hist)
 prompt='Predict upcoming4seconds using ONLY supplied paststructuredobservations. No tools, names, fightmemories, files, web or futurefacts. No rawimages supplied. Return JSON {end_phase_probabilities:{'+','.join(x+':probability' for x in PHASES)+'},first_action_probabilities:{'+','.join(x+':probability' for x in ACTIONS)+'},rationale,uncertainties}. Eachdistribution finite [0,1] sums1. Firstaction means first NEW committed onset in futurewindow, excludesongoing alreadyactiveaction. control_change meaningfulclinch/groundtransition/escape; takedownentry committedshot/throw/trip; submissionentry newlyvisibleattack. Do not conflate noevent withunknown.\nPastthrough '+str(t)+'s. Futurewindow '+str([t,t+4])+'.\n'+json.dumps(inp)
 a=call(f'{cid}_pred_{t}_{cond}',prompt);(O/'predictions'/f'{cid}_{t}_{cond}.json').write_text(json.dumps(a,indent=2));print('PRED',cid,t,cond,flush=True)
def score():
 rows=[];norm=[]
 for cid in 'EFGH':
  hist=json.loads((O/'observations'/f'{cid}.json').read_text())
  for t in [4,8,12,16]:
   ref=json.loads((O/'references'/f'{cid}_{t}.json').read_text())
   for cond in protocol['conditions']:
    pred=json.loads((O/'predictions'/f'{cid}_{t}_{cond}.json').read_text())
    for target,classes,key,truth in [('phase',PHASES,'end_phase_probabilities',ref['end_phase']),('action',ACTIONS,'first_action_probabilities',ref['first_new_committed_action'])]:
     ps=pred[key];assert set(ps)==set(classes) and all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in ps.values());total=sum(ps.values());assert abs(total-1)<=.02;probs={k:v/total for k,v in ps.items()};norm.append({'clip':cid,'cutoff':t,'condition':cond,'target':target,'sum_before':total});top=max(probs,key=probs.get);eligible=truth!='unclear';brier=sum((probs[k]-(k==truth))**2 for k in classes) if eligible else None
     rows.append({'clip':cid,'bout_group':manifest[cid]['source'],'cutoff':t,'condition':cond,'target':target,'truth':truth,'prediction':top,'eligible':eligible,'correct':top==truth if eligible else None,'brier':brier,'probabilities':probs})
   for target,truth,guess in [('phase',ref['end_phase'],next(a['end_phase'] for a in hist if a['end']==t)),('action',ref['first_new_committed_action'],'none')]:
    eligible=truth!='unclear';rows.append({'clip':cid,'bout_group':manifest[cid]['source'],'cutoff':t,'condition':'persistence' if target=='phase' else 'always_none','target':target,'truth':truth,'prediction':guess,'eligible':eligible,'correct':guess==truth if eligible else None,'brier':0 if eligible and guess==truth else 2 if eligible else None})
 summaries=[]
 for cond in protocol['conditions']+['persistence','always_none']:
  for tar in ['phase','action']:
   rs=[r for r in rows if r['condition']==cond and r['target']==tar and r['eligible']]
   if rs:summaries.append({'condition':cond,'target':tar,'n':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'mean_brier':sum(r['brier'] for r in rs)/len(rs)})
 (O/'case_scores.json').write_text(json.dumps(rows,indent=2));(O/'scores.json').write_text(json.dumps(summaries,indent=2));(O/'normalization.json').write_text(json.dumps(norm,indent=2));print('SCORED',summaries,flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(label,'EFGH'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(reference,[(c,t) for c in 'EFGH' for t in [4,8,12,16]]))
 (O/'frozen_references.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((O/'references').glob('*.json'))},indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(predict,[(c,t,cond) for c in 'EFGH' for t in [4,8,12,16] for cond in protocol['conditions']]))
 score()
