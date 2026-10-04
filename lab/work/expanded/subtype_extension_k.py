import sys,json,cv2,hashlib,math,time,concurrent.futures
from pathlib import Path
from PIL import Image,ImageDraw
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import subtype_study as s
O=R/'outputs/expanded/key_moments/subtypes/extension_K';O.mkdir(exist_ok=True);s.O=O;s.base.O=O;s.CASES=[('K',t) for t in [4,6,7]]
for f in ['audit','evidence','references','predictions','prefixes']:(O/f).mkdir(exist_ok=True)
proto={**s.PROTO,'cases':s.CASES,'extension':'Additionalselectedgroundcontrol-stress episode; separatefromregistered19caseexperiment. NewboutPHzT9Wl8HYQ;Kcutoffscorrelatedsameepisode.','created_unix':time.time()};(O/'protocol.json').write_text(json.dumps(proto,indent=2))
def prefix(c,t):
 src=R/'outputs/expanded/key_moments/control_transition/observations/K.json';hist=[a for a in json.loads(src.read_text()) if a['end']<=t];assert hist and max(a['end'] for a in hist)==t;dest=O/'prefixes'/f'{c}_{t}.json';dest.write_text(json.dumps(hist,indent=2));return hist,{'source':str(src.relative_to(R)),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'selection':'Filteredstrictcausalhistending<=cutoff; no future reference supplied'}
def evidence(c,t):
 path=R/'outputs/expanded/key_moments/control_transition/K.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(5);n=int(cap.get(7));width=int(cap.get(3));height=int(cap.get(4));times=[t-.5,t-.25]+[t+j/12 for j in range(48)]+[min(t+4,(n-1)/fps)];samples=[];images=[]
 for j,when in enumerate(times):
  ix=max(0,min(n-1,int(math.floor(when*fps+1e-8))));cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,im=cap.read();assert ok;actual=ix/fps;scope='referenceonly_preroll' if actual<t else 'endpoint_context' if j==len(times)-1 else 'target_horizon';assert actual<=t+4;samples.append({'native_frame_index':ix,'actual_time_seconds':actual,'requested_time_seconds':when,'scope':scope})
  if j%4==0:sheet=Image.new('RGB',(width*2,(height+28)*2),'#10171c');draw=ImageDraw.Draw(sheet)
  x=(j%4)%2*width;y=(j%4)//2*(height+28);sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+8,y+height+5),f'{c} actual t={actual:.5f}s {scope}',fill='white')
  if j%4==3 or j==len(times)-1:
   p=O/'evidence'/f'{c}_{t}_{j//4:02d}.jpg';sheet.save(p,quality=97);images.append(p)
 cap.release();return images,{'clip':c,'cutoff':t,'fps':fps,'decoded_native_dimensions':[width,height],'requested_sampling_hz':12,'source_path':str(path.relative_to(R)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':samples}
s.prefix=prefix;s.evidence=evidence
if __name__=='__main__':
 for c,t in s.CASES:prefix(c,t)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(s.ref,s.CASES))
 (O/'frozen_references.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((O/'references').glob('*.json'))},indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(s.pred,[(c,t) for c,t in s.CASES if (O/'references'/f'{c}_{t}.json').exists()]))
 rows=[];summary=[]
 for c,t in s.CASES:
  rp=O/'references'/f'{c}_{t}.json';pp=O/'predictions'/f'{c}_{t}.json'
  if not rp.exists() or not pp.exists():continue
  ref=json.loads(rp.read_text());pred=json.loads(pp.read_text());hist,_=prefix(c,t)
  for cond in ['rich_history_textonly','always_yes','always_no','uninformative_0.5']:
   for target in s.TARGETS:
    truth=ref['targets'][target]['occurrence'];p=pred['probabilities'][target] if cond=='rich_history_textonly' else 1 if cond=='always_yes' else 0 if cond=='always_no' else .5;y=1 if truth=='yes' else 0 if truth=='no' else None;rows.append({'clip':c,'cutoff':t,'past_phase':hist[-1]['end_phase'],'bout_group':'PHzT9Wl8HYQ','subset':'selected_control_stress_test','condition':cond,'target':target,'truth':truth,'probability':p,'eligible':y is not None,'brier':(p-y)**2 if y is not None else None,'alert':p>=.5,'onsets_seconds':ref['targets'][target]['onsets_seconds']})
 for cond in ['rich_history_textonly','always_yes','always_no','uninformative_0.5']:
  for target in s.TARGETS:
   allrs=[r for r in rows if r['condition']==cond and r['target']==target];rs=[r for r in allrs if r['eligible']];pos=sum(r['truth']=='yes' for r in rs);neg=sum(r['truth']=='no' for r in rs);tp=sum(r['alert'] and r['truth']=='yes' for r in rs);fp=sum(r['alert'] and r['truth']=='no' for r in rs)
   if allrs:summary.append({'subset':'selected_control_stress_test','past_phase':'ground','condition':cond,'target':target,'windows':len(allrs),'eligible':len(rs),'positives':pos,'negatives':neg,'unclear':len(allrs)-len(rs),'mean_brier':sum(r['brier'] for r in rs)/len(rs) if rs else None,'true_alerts':tp,'false_alerts':fp,'missed_positives':pos-tp,'true_negatives':neg-fp,'accuracy_at_0_5':(tp+neg-fp)/len(rs) if rs else None,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/pos if pos else None,'false_alarm_rate':fp/neg if neg else None})
 (O/'case_scores.json').write_text(json.dumps(rows,indent=2));(O/'scores.json').write_text(json.dumps(summary,indent=2));(O/'coverage.json').write_text(json.dumps({'registered_cases':3,'references_completed':len(list((O/'references').glob('*.json'))),'forecasts_completed':len(list((O/'predictions').glob('*.json'))),'status':'complete' if len(rows)==72 else 'incomplete','selection':'Threecutoffssameoneepisode'},indent=2));print('COMPLETE',flush=True)
