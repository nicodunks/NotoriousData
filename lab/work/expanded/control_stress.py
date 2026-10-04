import json,sys,time,hashlib,concurrent.futures
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as b
O=R/'outputs/expanded/key_moments/control_transition';b.O=O;b.manifest['K']={'clip':str((O/'K.mp4').relative_to(R))}
for d in ['audit','evidence','observations','references','predictions']:(O/d).mkdir(exist_ok=True)
TARGETS=['strike_attempt','takedown_attempt','submission_entry','major_control_transition'];CONDS=['phase_recent_events','rich_history']
DEFS='strike_attempt: newly committed punch/kick/knee/elbow including ground strikes regardless contact, exclude feints/ongoing single strike. takedown_attempt: newly committed shot/trip/throw even defended. submission_entry: newly visible committed submission attack, notordinary grip. major_control_transition: newly established meaningful pass/reversal/escape/dominant control, excludes microgrips and mere takedown descent.'
P={'model':'gpt-6.1-sol','reasoning':'high','training':False,'cutoffs':[4,6,7],'horizon':4,'source_start':1531,'selection':'Retrospective control-transition stress test selected fromscoutS60 after aroll was seen; not heldout. Three nearbycutoffs sameepisode, separatefromordinaryE-Hcases.','causality':'Eachprefixcall seesonlysuppliedprecutoffframes+earlierprefixlabels. Predictorsseeonlystructuredprefixes noimages/future/names.','target_definitions':DEFS,'reference':'Independent4Hzfuture +referenceonlypreroll; unknownunscored; laternativeverify separately','conditions':CONDS,'targets':TARGETS,'probabilities':'independentBernoulli notsum1'};(O/'protocol.json').write_text(json.dumps(P,indent=2))
hist=[]
for st,en in [(0,4),(4,6),(6,7)]:
 ims=b.evidence('K',st)[:en-st]
 a=b.call(f'K_obs_{st}_{en}',b.RULE+'\nLabel ONLY ['+str(st)+','+str(en)+'), exclusiveend. Suppliedfullframes4Hz. Earlier CAUSAL labels:'+json.dumps(hist)+'\n'+b.LABEL,ims);a['start']=st;a['end']=en;hist.append(a);(O/'observations/K.json').write_text(json.dumps(hist,indent=2));print('OBS',st,en,flush=True)

def ref(t):
 prompt=b.RULE+'\nIndependent keyevent reference ONLY ['+str(t)+','+str(t+4)+'). '+DEFS+'\nANY newonsetanywhereinsidehorizon, notfirstaction. Prerollreferenceonly, excludealreadyongoingaction. Endpointboundaryoutsideonsettarget. Groundrollmereinitiation may precede newlyestablishedcontrol; recordwhichobservedtime supports target. Return JSON {start,end,targets:{'+','.join(k+':{occurrence:yes|no|unclear,onsets_seconds:[],actors:[],evidence,visibility_limits}' for k in TARGETS)+'},uncertainties}. Multiplecategoriescanbepositive. No requiresconvincingvisibility, otherwiseunclear. Excludenames/outcomememories.'
 a=b.call(f'K_ref_{t}',prompt,b.reference_evidence('K',t));(O/'references'/f'K_{t}.json').write_text(json.dumps(a,indent=2));print('REF',t,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(ref,[4,6,7]))
(O/'frozen_references.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'references').glob('*.json')},indent=2))

def pred(case):
 t,cond=case;past=[x for x in hist if x['end']<=t];assert all(x['end']<=t for x in past)
 if cond=='phase_recent_events':past=[{k:x[k] for k in ['start','end','end_phase','events','visibility'] if k in x} for x in past[-1:]]
 prompt='Predict ANY keyeventonset in ['+str(t)+','+str(t+4)+') ONLY frompaststructuredannotations through '+str(t)+'. No rawimages/names/future/files/tools/web/fightmemory. '+DEFS+'\nIndependent Bernoulli0–1probabilities, doNOTsumto1. Return JSON {probabilities:{'+','.join(k+':probability' for k in TARGETS)+'},rationale_by_target:{'+','.join(k+':reason' for k in TARGETS)+'},uncertainties}.\n'+json.dumps({'past_observations':past})
 a=b.call(f'K_pred_{t}_{cond}',prompt);(O/'predictions'/f'K_{t}_{cond}.json').write_text(json.dumps(a,indent=2));print('PRED',t,cond,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(pred,[(t,c) for t in [4,6,7] for c in CONDS]))
rows=[]
for t in [4,6,7]:
 ref=json.load(open(O/'references'/f'K_{t}.json'))
 for cond in CONDS+['always_no','uninformative_0.5']:
  a=json.load(open(O/'predictions'/f'K_{t}_{cond}.json')) if cond in CONDS else None
  for k in TARGETS:
   tr=ref['targets'][k]['occurrence'];p=a['probabilities'][k] if a else 0 if cond=='always_no' else .5;assert 0<=p<=1
   rows.append({'clip':'K','cutoff':t,'condition':cond,'target':k,'truth':tr,'probability':p,'brier':(p-(tr=='yes'))**2 if tr!='unclear' else None,'onsets_seconds':ref['targets'][k]['onsets_seconds']})
(O/'case_scores.json').write_text(json.dumps(rows,indent=2));print('SCORED',flush=True)
