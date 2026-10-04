"""Exploratory available-case preview; separate from full frozen study; no invented results."""
import importlib.util,json,shutil,concurrent.futures,hashlib,math,time
from pathlib import Path
R=Path.cwd();sp=importlib.util.spec_from_file_location('causal','work/expanded/causal_v2.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);original=m.O;m.O=R/'outputs/expanded/causal_preview'
for d in ['evidence','observations','references','predictions','audit']:(m.O/d).mkdir(parents=True,exist_ok=True)
cases=[('E',4),('E',8),('F',8),('F',16)]
for c in ['E','F']:shutil.copy2(original/'observations'/f'{c}.json',m.O/'observations'/f'{c}.json')
(m.O/'protocol.json').write_text(json.dumps({'scope':'Exploratory four available cases, selected after main pipeline transport delays. Not full-study evidence or statistical proof. Separate outputs.','cases':cases,'conditions':m.protocol['conditions'],'model':'gpt-6.1-sol','reasoning':'high','causality':'Only pre-cutoff structured observations and pose in forecasts. Reference-only pre-roll and endpoint supplied separately.','input_observation_hashes':{c:hashlib.sha256((m.O/'observations'/f'{c}.json').read_bytes()).hexdigest() for c in ['E','F']},'started_unix':time.time()},indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(m.reference,cases))
(m.O/'frozen_references.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (m.O/'references').glob('*.json')},indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(m.predict,[(c,t,cond) for c,t in cases for cond in m.protocol['conditions']]))
rows=[]
for c,t in cases:
 ref=json.loads((m.O/'references'/f'{c}_{t}.json').read_text());hist=json.loads((m.O/'observations'/f'{c}.json').read_text())
 for cond in m.protocol['conditions']:
  p=json.loads((m.O/'predictions'/f'{c}_{t}_{cond}.json').read_text())
  for target,key,classes,truth in [('phase','end_phase_probabilities',m.PHASES,ref['end_phase']),('action','first_action_probabilities',m.ACTIONS,ref['first_new_committed_action'])]:
   probs=p[key];assert set(probs)==set(classes);assert all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in probs.values());su=sum(probs.values());assert abs(su-1)<=.02;probs={k:v/su for k,v in probs.items()};top=max(probs,key=probs.get);rows.append({'clip':c,'cutoff':t,'condition':cond,'target':target,'truth':truth,'prediction':top,'scoreable':truth!='unclear','correct':top==truth if truth!='unclear' else None,'brier':sum((probs[k]-(k==truth))**2 for k in classes) if truth!='unclear' else None,'sum_before_normalization':su,'probabilities':probs})
 for target,truth,guess in [('phase',ref['end_phase'],next(a['end_phase'] for a in hist if a['end']==t)),('action',ref['first_new_committed_action'],'none')]:
  rows.append({'clip':c,'cutoff':t,'condition':'persistence' if target=='phase' else 'always_none','target':target,'truth':truth,'prediction':guess,'scoreable':truth!='unclear','correct':guess==truth if truth!='unclear' else None,'brier':0 if guess==truth else 2 if truth!='unclear' else None})
(m.O/'case_scores.json').write_text(json.dumps(rows,indent=2));print('PREVIEW COMPLETE',len(rows),flush=True)
