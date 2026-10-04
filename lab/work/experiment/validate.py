import json,hashlib,math
from pathlib import Path
O=Path.cwd()/'outputs/experiment';bad=[];manifest=[];normalizations=[]
expected={'phase_probs':{'standing','clinch','ground','unclear'},'event_probs':{'strike_attempt','takedown_attempt','ground_position_change'},'next_action_probs':{'punch','kick_knee_elbow','takedown_entry','ground_strike','ground_control_change','none','unclear'}}
for p in sorted((O/'predictions').glob('*.json')):
 x=json.loads(p.read_text());meta=json.loads((O/'audit'/f'{p.stem}.metadata.json').read_text());prompt=(O/'audit'/f'{p.stem}.prompt.txt').read_text()
 assert hashlib.sha256(prompt.encode()).hexdigest()==meta['prompt_sha256'];assert meta['model']=='gpt-6.1-sol' and meta['reasoning']=='high' and not meta['tools_used'];assert not meta['images']
 for k,keys in expected.items():
  assert set(x[k])==keys,(p,k,x[k]);assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in x[k].values())
  if k!='event_probs':
   total=sum(x[k].values());assert total>0 and abs(total-1)<=.020001,(p,k,total)
   if abs(total-1)>1e-6:normalizations.append(dict(path=str(p.relative_to(O)),field=k,raw_sum=total,policy='divide each categorical probability by raw sum; raw forecast is unchanged'))
 manifest.append(dict(path=str(p.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),prompt_sha256=meta['prompt_sha256'],model=meta['model']))
refs=[]
for c in 'ABCD':
 for t in range(0,20,4):
  p=O/'adjudication'/f'{c}_{t:02d}.json';p=p if p.exists() else O/'verification'/f'{c}_{t:02d}.json'
  x=json.loads(p.read_text());assert x['start']==t and x['end']==t+4
  assert x['end_phase'] in expected['phase_probs'];assert x.get('first_action','unclear') in expected['next_action_probs'];assert set(x['event_flags'])==expected['event_probs'];assert all(v in ['yes','no','unclear'] for v in x['event_flags'].values())
  refs.append(dict(path=str(p.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(O/'frozen_scoring_manifest.json').write_text(json.dumps(dict(predictions=manifest,references=refs,normalization='Categorical sums within 0.02 of one are rescaled to one in scoring/display; raw forecasts unchanged',reference_kind='independent GPT-6.1 Sol visual review, with documented adjudication where present; no human expert ground truth'),indent=2))
(O/'probability_normalization.json').write_text(json.dumps(normalizations,indent=2))
print('Validated',len(manifest),'predictions and',len(refs),'reference intervals.')
