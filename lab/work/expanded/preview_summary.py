import json,hashlib
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/causal_preview';rows=json.loads((O/'case_scores.json').read_text());summ=[];paired=[]
for cond in ['coarse_history','rich_history','rich_pose_history','persistence','always_none']:
 for target in ['phase','action']:
  rs=[r for r in rows if r['condition']==cond and r['target']==target and r['scoreable']]
  if rs:summ.append({'condition':cond,'target':target,'scoreable_cases':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'mean_categorical_brier':sum(r['brier'] for r in rs)/len(rs)})
for c,t in [('E',4),('E',8),('F',8),('F',16)]:
 for target in ['phase','action']:
  rs={r['condition']:r for r in rows if r['clip']==c and r['cutoff']==t and r['target']==target and r['scoreable']}
  if not rs:continue
  for cond in ['rich_history','rich_pose_history']:
   paired.append({'clip':c,'cutoff':t,'target':target,'condition':cond,'comparator':'coarse_history','brier_delta':rs[cond]['brier']-rs['coarse_history']['brier'],'negative_means_lower_error':True})
(O/'scores.json').write_text(json.dumps(summ,indent=2));(O/'paired_deltas.json').write_text(json.dumps(paired,indent=2))
inputs=[]
for p in (O/'audit').glob('*.metadata.json'):
 m=json.loads(p.read_text());assert m['model']=='gpt-6.1-sol' and not m['tools_used'];prompt=p.with_name(p.name.replace('.metadata.json','.prompt.txt'));assert hashlib.sha256(prompt.read_bytes()).hexdigest()==m['prompt_sha256']
 for im in m['images']:assert hashlib.sha256((R/im['path']).read_bytes()).hexdigest()==im['sha256']
 inputs.append(p.name)
for p in (O/'predictions').glob('*.json'):
 a=json.load(open(p));assert a['end_phase_probabilities'] and a['first_action_probabilities']
(O/'validation.json').write_text(json.dumps({'status':'checked','model_calls':len(inputs),'predictions':12,'references':4,'checks':['Prompt and image hashes match actualsuppliedfiles','Tool-free model metadata','Finite distribution ranges and categoricalnormalization verified by scoring','Four cases only; two unclear actionreferences excluded','Future images appear only in referencecalls; predictors have no images and pre-cutoff structuredinputs'],'limitations':'This does not establish expert ground truth or general predictive skill.'},indent=2));print(summ)
