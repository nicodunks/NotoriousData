import sys,json,cv2,hashlib,time,concurrent.futures,math
from pathlib import Path
from PIL import Image,ImageDraw
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as base
O=R/'outputs/expanded/key_moments/subtypes';O.mkdir(exist_ok=True);base.O=O
for f in ['audit','evidence','references','predictions','prefixes']:(O/f).mkdir(exist_ok=True)
TARGETS=['punch_attempt','kick_attempt','knee_elbow_attempt','ground_strike_attempt','takedown_attempt','submission_entry']
CASES=[(c,t) for c in 'EFGH' for t in [4,8,12,16]]+[('B',t) for t in [4,6,7]]
DEFS='punch_attempt: newly committed punch while actor/opponent exchange is standing or in standing clinch. kick_attempt: newly committed kick in standing or standingclinch. knee_elbow_attempt: newly committed knee or elbow in standing or standingclinch. ground_strike_attempt: any newly committed punch/kick/knee/elbow in a ground exchange; exclude it from the other3striking categories. If phase at onset is ambiguous, mark relevant target unclear rather than doublecount. takedown_attempt: newly committed shot/trip/throw even if defended. submission_entry: newly visible established submission attack, not ordinary grip/control. Each is occurrence of ANYNEW committed onset within horizon, not first action. Feints/probes excluded, alreadyongoing individual action atcutoff excluded, reinitiated strikes after reset count. Contact/clean/block/damage irrelevant to attempt.'
from datetime import datetime,timezone
STOP=datetime(2026,10,2,14,22,tzinfo=timezone.utc).timestamp()
PROTO={'model':'gpt-6.1-sol','reasoning':'high','cases':CASES,'targets':TARGETS,'definitions':DEFS,'horizon_seconds':4,'condition':'rich_history_textonly','training':False,'predictor':'Only existing strictcausalprefixes; no images, otheranswers, fightnames orfuturefacts.','references':'Independent new12Hzfullframes preservingdecodednativeclip resolution;0.5sreferenceonlypreroll+endphasecontext; actualfloor-frame timestamps; model mayresize largepages. No oldreferences/predictions provided.','metrics':'ConventionalbinaryBrier,0.5alerts,positive/negative/unclear counts pertarget andPASTphase; alwaysyes/no/0.5constants.','dependence':'E/F samebout; B/G samebout; Hthirdbout. B4/6/7sameoneepisode, intentionallyselectedstresscase. Allneighboringwindows correlated; notheldoutcalibration.','stop_new_launch_utc':'2026-10-02T14:22:00Z','created_unix':time.time()}
if not (O/'protocol.json').exists():(O/'protocol.json').write_text(json.dumps(PROTO,indent=2))
def can_launch():return time.time()<STOP
def prefix(c,t):
 path=R/'outputs/expanded/key_moments/prefixes'/f'{c}_{t}.json';hist=json.loads(path.read_text());assert all(x['end']<=t for x in hist)
 dest=O/'prefixes'/f'{c}_{t}.json';dest.write_text(json.dumps(hist,indent=2));return hist,{'source':str(path.relative_to(R)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
def evidence(c,t):
 path=R/('outputs/experiment/clips/B.mp4' if c=='B' else f'outputs/expanded/clips/{c}.mp4');cap=cv2.VideoCapture(str(path));fps=cap.get(5);n=int(cap.get(7));width=int(cap.get(3));height=int(cap.get(4));times=[t-.5,t-.25]+[t+j/12 for j in range(48)]+[min(t+4,(n-1)/fps)];samples=[];images=[]
 for j,when in enumerate(times):
  ix=max(0,min(n-1,int(math.floor(when*fps+1e-8))));cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,im=cap.read();assert ok;actual=ix/fps
  scope='referenceonly_preroll' if actual<t else 'endpoint_context' if j==len(times)-1 else 'target_horizon'
  assert actual<=t+4
  samples.append({'native_frame_index':ix,'actual_time_seconds':actual,'requested_time_seconds':when,'scope':scope})
  if j%4==0:sheet=Image.new('RGB',(width*2,(height+28)*2),'#10171c');draw=ImageDraw.Draw(sheet)
  x=(j%4)%2*width;y=(j%4)//2*(height+28);sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+8,y+height+5),f'{c} actual t={actual:.5f}s {scope}',fill='white')
  if j%4==3 or j==len(times)-1:
   q=O/'evidence'/f'{c}_{t}_{j//4:02d}.jpg';sheet.save(q,quality=97);images.append(q)
 cap.release();return images,{'clip':c,'cutoff':t,'fps':fps,'decoded_native_dimensions':[width,height],'requested_sampling_hz':12,'source_path':str(path.relative_to(R)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':samples}
def ref(case):
 c,t=case
 if not can_launch():print('DEADLINE_SKIP_REF',c,t,flush=True);return
 images,man=evidence(c,t);(O/'evidence'/f'{c}_{t}.sampling.json').write_text(json.dumps(man,indent=2))
 prompt=base.RULE+'\nIndependent subtype occurrence reference for target horizon ['+str(t)+','+str(t+4)+'). '+DEFS+'\nReview allcategories independently ANYWHERE in window. Images12Hz preserving decodedclip nativepixels in2x2pages; frontend mayresize. Actualtimestampsprinted. Preroll forongoing-actioncontext only; endpointcontext nevercounts a new onset at/afterend. No previousreferenceanswers orpredictions supplied. No label desires supplied. For eachtarget return yes|no|unclear. A no requires adequateobservablecoverage, otherwiseunclear. Return JSON {start,end,targets:{'+','.join(k+':{occurrence:yes|no|unclear,onsets_seconds:[],actors:[],phase_at_onset:[],evidence,visibility_limits}' for k in TARGETS)+'},uncertainties}. Clothingcolors only. Do not infer hiddenmovement, contact, outcomes orfighter names.'
 a=base.call(f'{c}_ref_{t}',prompt,images)
 for k in TARGETS:assert a['targets'][k]['occurrence'] in ['yes','no','unclear']
 (O/'references'/f'{c}_{t}.json').write_text(json.dumps(a,indent=2));print('REF',c,t,flush=True)
def pred(case):
 c,t=case
 if not can_launch():print('DEADLINE_SKIP_PRED',c,t,flush=True);return
 hist,lineage=prefix(c,t);(O/'prefixes'/f'{c}_{t}.lineage.json').write_text(json.dumps(lineage,indent=2))
 prompt='Forecast eachindependentkeyevent occurrence ANYWHERE in next4seconds ONLYfrom suppliedpaststructuredannotations. No tools,rawimages,othermodelanswers,names,fightmemory,web,futurefacts. '+DEFS+' EachindependentBernoulli probability0to1; NOTnormalizedacrosscategories. ReturnJSON {probabilities:{'+','.join(k+':probability' for k in TARGETS)+'},rationale_by_target:{'+','.join(k+':reason' for k in TARGETS)+'},uncertainties}. Futurehorizon ['+str(t)+','+str(t+4)+'); causalpastthrough '+str(t)+'seconds:\n'+json.dumps(hist)
 a=base.call(f'{c}_pred_{t}_rich_history',prompt)
 assert set(a['probabilities'])==set(TARGETS)
 for k in TARGETS:assert isinstance(a['probabilities'][k],(int,float)) and math.isfinite(a['probabilities'][k]) and 0<=a['probabilities'][k]<=1
 (O/'predictions'/f'{c}_{t}.json').write_text(json.dumps(a,indent=2));print('PRED',c,t,flush=True)
def score():
 rows=[];coverage=[]
 for c,t in CASES:
  rp=O/'references'/f'{c}_{t}.json';pp=O/'predictions'/f'{c}_{t}.json'
  if not rp.exists():coverage.append({'clip':c,'cutoff':t,'reference':'missing','forecast':'missing' if not pp.exists() else 'exists'});continue
  ref=json.loads(rp.read_text());hist,_=prefix(c,t);phase=hist[-1]['end_phase'];pred=json.loads(pp.read_text()) if pp.exists() else None;coverage.append({'clip':c,'cutoff':t,'reference':'complete','forecast':'complete' if pred else 'missing','past_phase':phase})
  for condition in ['rich_history_textonly','always_yes','always_no','uninformative_0.5']:
   if condition=='rich_history_textonly' and pred is None:continue
   for k in TARGETS:
    truth=ref['targets'][k]['occurrence'];p=pred['probabilities'][k] if condition=='rich_history_textonly' else 1 if condition=='always_yes' else 0 if condition=='always_no' else .5;y=1 if truth=='yes' else 0 if truth=='no' else None
    rows.append({'clip':c,'cutoff':t,'bout_group':'pereira-adesanya' if c in 'EF' else 'khabib-mcgregor' if c in 'GB' else 'oliveira-poirier','subset':'selected_takedown_stress_test' if c=='B' else 'ordinary_registered','past_phase':phase,'condition':condition,'target':k,'truth':truth,'probability':p,'eligible':y is not None,'brier':(p-y)**2 if y is not None else None,'alert':p>=.5,'onsets_seconds':ref['targets'][k]['onsets_seconds']})
 summaries=[]
 for subset in ['ordinary_registered','selected_takedown_stress_test']:
  for phase in ['all','standing','clinch','ground','unclear']:
   for condition in ['rich_history_textonly','always_yes','always_no','uninformative_0.5']:
    for k in TARGETS:
     allrs=[r for r in rows if r['subset']==subset and (phase=='all' or r['past_phase']==phase) and r['condition']==condition and r['target']==k];rs=[r for r in allrs if r['eligible']]
     if not allrs:continue
     pos=sum(r['truth']=='yes' for r in rs);neg=sum(r['truth']=='no' for r in rs);tp=sum(r['alert'] and r['truth']=='yes' for r in rs);fp=sum(r['alert'] and r['truth']=='no' for r in rs)
     summaries.append({'subset':subset,'past_phase':phase,'condition':condition,'target':k,'windows':len(allrs),'eligible':len(rs),'positives':pos,'negatives':neg,'unclear':len(allrs)-len(rs),'mean_brier':sum(r['brier'] for r in rs)/len(rs) if rs else None,'true_alerts':tp,'false_alerts':fp,'missed_positives':pos-tp,'true_negatives':neg-fp,'accuracy_at_0_5':(tp+neg-fp)/len(rs) if rs else None,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/pos if pos else None,'false_alarm_rate':fp/neg if neg else None})
 (O/'case_scores.json').write_text(json.dumps(rows,indent=2));(O/'scores.json').write_text(json.dumps(summaries,indent=2));(O/'coverage.json').write_text(json.dumps({'registered_cases':19,'references_completed':sum(r['reference']=='complete' for r in coverage),'forecasts_completed':sum(r.get('forecast')=='complete' for r in coverage),'status':'complete' if len(coverage)==19 and all(r.get('forecast')=='complete' for r in coverage) else 'incomplete','cases':coverage},indent=2));print('SCORED',len(rows),flush=True)
if __name__=='__main__':
 for c,t in CASES:prefix(c,t)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(ref,CASES))
 (O/'frozen_references.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((O/'references').glob('*.json'))},indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(pred,[(c,t) for c,t in CASES if (O/'references'/f'{c}_{t}.json').exists()]))
 score();print('COMPLETE',flush=True)
