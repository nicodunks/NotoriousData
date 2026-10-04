"""Keyevent occurrence, frozen causal prefixes, independent future review; no training."""
import sys,json,hashlib,time,concurrent.futures,math
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as base
sys.path.insert(0,str(R/'work/experiment'));from colors import colorize
O=R/'outputs/expanded/key_moments';O.mkdir(exist_ok=True);base.O=O
for d in ['audit','evidence','references','predictions','prefixes']:(O/d).mkdir(exist_ok=True)
base.manifest['B']={'clip':'outputs/experiment/clips/B.mp4','source':'khabib-mcgregor'}
TARGETS=['strike_attempt','takedown_attempt','submission_entry','major_control_transition']
CONDITIONS=['phase_recent_events','rich_history','rich_pose_history']
CASES=[(c,t) for c in 'EFGH' for t in [4,8,12,16]]+[('B',t) for t in [4,6,7]]
DEFS='strike_attempt: a newly committed punch/kick/knee/elbow including ground strikes, whether it lands or misses. Feints/probes alone excluded. Reinitiated strike after reset counts; a single strike already underway at cutoff does not. takedown_attempt: newly committed shot/trip/throw even if defended. submission_entry: newly visible committed submission attack, not ordinary grip/control. major_control_transition: newly established meaningful pass/reversal/escape/dominant control; exclude microgrips and do not automatically double-count takedown descent as separate control transition.'
P={'model':'gpt-6.1-sol','reasoning':'high','targets':TARGETS,'definitions':DEFS,'horizon_seconds':4,'cases':CASES,'conditions':CONDITIONS,'probabilities':'Independent Bernoulli; do NOT sum to1','causality':'Predictors see only annotations ending<=cutoff and optional poses captured<cutoff. No images, future labels, names, fight outcomes, external tools.','selection':'E-H previously registered ordinary windows; B4/6/7 deliberately selected retrospective failure stress test, same one takedown episode. Dependentwindows in3 bouts. Not heldout performance or population calibration.','references':'Independent new4Hz fullfuture review +0.5s reference-only preroll; no prediction inputs. Unknown unscored. Native-rate revisit flagged uncertain keyevents if feasible.','metrics':'Conventional binary Brier (p-y)^2, pertarget counts, frozen0.5 alert threshold, alwaysno/0.5/recent-event persistence, pairedcase deltas. B episode warning curve separately, not3 independent positives.','training':False,'created_unix':time.time()}
(O/'protocol.json').write_text(json.dumps(P,indent=2))

def prefix(c,t):
 if c!='B':hist=[x for x in json.loads((R/'outputs/expanded/causal_v2/observations'/f'{c}.json').read_text()) if x['end']<=t]
 else:
  hist=[colorize(json.loads((R/'outputs/experiment/labels/B.json').read_text())[0])]
  if t>=6:hist.append(json.loads((R/'outputs/experiment/diagnostic/past_4_6.json').read_text()))
  if t>=7:hist.append(json.loads((R/'outputs/experiment/diagnostic/past_6_7.json').read_text()))
 assert all(x['end']<=t for x in hist)
 (O/'prefixes'/f'{c}_{t}.json').write_text(json.dumps(hist,indent=2));return hist

def ref(case):
 c,t=case;prompt=base.RULE+'\nIndependent keyevent reference for ['+str(t)+','+str(t+4)+'). '+DEFS+'\nReview ANY NEW onset anywhere in this horizon, not just first action. Multiple target categories may be positive. Preroll is reference-only; distinguish already underway actions. Endpoint excluded as NEWonset; context only. Return JSON {start,end,targets:{'+','.join(k+':{occurrence:yes|no|unclear,onsets_seconds:[],actors:[],evidence,visibility_limits}' for k in TARGETS)+'},uncertainties}. No requires convincing visibility across whole window, otherwise unclear. Do not infer contact, damage or outcomes. All temporal claims approximate sampling limits.'
 a=base.call(f'{c}_ref_{t}',prompt,base.reference_evidence(c,t))
 for k in TARGETS:assert a['targets'][k]['occurrence'] in ['yes','no','unclear']
 (O/'references'/f'{c}_{t}.json').write_text(json.dumps(a,indent=2));print('REF',c,t,flush=True)

def bpose(t):
 obj=json.loads((R/'outputs/experiment/body_data/B_rtmw_133.json').read_text());rows=obj.get('frames',obj) if isinstance(obj,dict) else obj
 return {'ownership':'unverified, potentially merged estimates; rawscores notprobabilities','frames':[r for r in rows if r.get('time',r.get('capture_time_seconds',r.get('t',999)))<t][-8:]}

def pred(case):
 c,t,cond=case;hist=prefix(c,t)
 if cond=='phase_recent_events':hist=[{k:x[k] for k in ['start','end','end_phase','events','event_flags','visibility'] if k in x} for x in hist[-1:]]
 inp={'past_observations':hist}
 if cond=='rich_pose_history':inp['past_pose']=base.pose_prefix(c,t) if c!='B' else bpose(t)
 prompt='Predict keyevent occurrence in upcoming ['+str(t)+','+str(t+4)+') ONLY from provided causal annotated sequence. No tools, rawframes, names, fightmemories, web, future facts. '+DEFS+'\nEach target independent Bernoulli probability0to1; probabilities MUST NOT sum1 requirement. Return JSON {probabilities:{'+','.join(k+':probability' for k in TARGETS)+'},rationale_by_target:{'+','.join(k+':reason' for k in TARGETS)+'},uncertainties}. Consider any onset anywhere in4seconds, not first action. Preserve uncertainty; do not treat unknown as confirmed absence. Past data through '+str(t)+'s:\n'+json.dumps(inp)
 a=base.call(f'{c}_pred_{t}_{cond}',prompt)
 for k in TARGETS:assert isinstance(a['probabilities'][k],(int,float)) and math.isfinite(a['probabilities'][k]) and 0<=a['probabilities'][k]<=1
 (O/'predictions'/f'{c}_{t}_{cond}.json').write_text(json.dumps(a,indent=2));print('PRED',c,t,cond,flush=True)

def score():
 rows=[]
 for c,t in CASES:
  ref=json.loads((O/'references'/f'{c}_{t}.json').read_text())
  for cond in CONDITIONS+['always_no','uninformative_0.5']:
   pred=json.loads((O/'predictions'/f'{c}_{t}_{cond}.json').read_text()) if cond in CONDITIONS else None
   for k in TARGETS:
    truth=ref['targets'][k]['occurrence'];p=pred['probabilities'][k] if pred else 0 if cond=='always_no' else .5;y=int(truth=='yes') if truth!='unclear' else None
    rows.append({'clip':c,'cutoff':t,'bout_group':base.manifest[c]['source'],'subset':'selected_takedown_stress_test' if c=='B' else 'ordinary_registered','condition':cond,'target':k,'truth':truth,'probability':p,'alert':p>=.5,'brier':(p-y)**2 if y is not None else None,'onsets_seconds':ref['targets'][k]['onsets_seconds']})
 sums=[]
 for subset in ['ordinary_registered','selected_takedown_stress_test']:
  for cond in CONDITIONS+['always_no','uninformative_0.5']:
   for k in TARGETS:
    rs=[r for r in rows if r['subset']==subset and r['condition']==cond and r['target']==k and r['brier'] is not None]
    if rs:sums.append({'subset':subset,'condition':cond,'target':k,'n':len(rs),'positives':sum(r['truth']=='yes' for r in rs),'mean_brier':sum(r['brier'] for r in rs)/len(rs),'positive_alerts':sum(r['alert'] and r['truth']=='yes' for r in rs),'false_alerts':sum(r['alert'] and r['truth']=='no' for r in rs),'missed_positives':sum(not r['alert'] and r['truth']=='yes' for r in rs)})
 (O/'case_scores.json').write_text(json.dumps(rows,indent=2));(O/'scores.json').write_text(json.dumps(sums,indent=2));print('SCORED',json.dumps(sums),flush=True)
if __name__=='__main__':
 for c,t in CASES:prefix(c,t)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(ref,CASES))
 (O/'frozen_references.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((O/'references').glob('*.json'))},indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(pred,[(c,t,k) for c,t in CASES for k in CONDITIONS]))
 score()
