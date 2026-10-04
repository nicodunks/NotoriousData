import json,hashlib,re,collections,statistics
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/key_moments';rows=json.load(open(O/'case_scores.json'));sups=json.load(open(O/'supplements/case_scores.json'));hist={}
for p in (O/'prefixes').glob('*.json'):hist[p.stem]=json.load(open(p))
for r in rows:r['past_phase']=hist[f"{r['clip']}_{r['cutoff']}"][-1]['end_phase']
for r in sups:
 r['past_phase']=hist[f"{r['clip']}_{r['cutoff']}"][-1]['end_phase'];r['subset']='selected_takedown_stress_test' if r['clip']=='B' else 'ordinary_registered'
for r in sups:
 if r.get('probability') is not None:r['alert']=r['probability']>=.5
rows+=sups
native={}
for p in (O/'native_review').glob('*.json'):
 try:a=json.load(open(p))
 except:continue
 if isinstance(a,dict) and 'targets' in a and 'case_id'in a:native[a['case_id']]=a
bc=json.load(open(O/'native_review/B_control.json'))
for x in bc['reviews']:
 key=f"B_{x['cutoff']}";native.setdefault(key,{'targets':{}})['targets']['major_control_transition']=x['major_control_transition']
versions={'frozen_sparse_reference':rows,'selective_native_review_sensitivity':[]}
changes=[]
for r in rows:
 x=dict(r);k=f"{r['clip']}_{r['cutoff']}";tr=native.get(k,{}).get('targets',{}).get(r['target'])
 if tr:
  x['truth']=tr['occurrence'];x['reference_basis']='later_targeted_native_review';x['brier']=(x['probability']-(x['truth']=='yes'))**2 if x['probability'] is not None and x['truth']!='unclear' else None
  if r['truth']!=x['truth']:changes.append({'case':k,'target':r['target'],'frozen':r['truth'],'native':x['truth']})
 else:x['reference_basis']='frozen_sparse_review'
 versions['selective_native_review_sensitivity'].append(x)
summary=[]
for ver,rr in versions.items():
 groups=collections.defaultdict(list)
 for r in rr:groups[(r['subset'],r['past_phase'],r['condition'],r['target'])].append(r)
 for (sub,phase,cond,tar),rs in groups.items():
  eligible=[r for r in rs if r['brier'] is not None];neg=[r for r in eligible if r['truth']=='no'];pos=[r for r in eligible if r['truth']=='yes']
  summary.append({'reference_version':ver,'subset':sub,'past_phase':phase,'condition':cond,'target':tar,'total_cells':len(rs),'reference_yes':sum(r['truth']=='yes' for r in rs),'reference_no':sum(r['truth']=='no' for r in rs),'reference_unclear':sum(r['truth']=='unclear' for r in rs),'scored_cells':len(eligible),'predictor_abstentions':sum(r.get('probability') is None for r in rs),'mean_brier':statistics.mean(r['brier'] for r in eligible) if eligible else None,'positive_alerts':sum(r['alert'] for r in pos),'missed_positives':sum(not r['alert'] for r in pos),'false_alerts':sum(r['alert'] for r in neg),'false_alert_rate':sum(r['alert'] for r in neg)/len(neg) if neg else None,'alert_accuracy':sum(r['alert']==(r['truth']=='yes') for r in eligible)/len(eligible) if eligible else None,'interpretation':'Missingness differs byphase/target; groupdependentcases descriptive only. No FPR withoutnegatives. Selectivenativereview notrandom sensitivityset.'})
(O/'stratified_scores.json').write_text(json.dumps(summary,indent=2));(O/'sensitivity_case_scores.json').write_text(json.dumps(versions['selective_native_review_sensitivity'],indent=2));(O/'reference_changes.json').write_text(json.dumps(list({json.dumps(x,sort_keys=True):x for x in changes}.values()),indent=2))
checks=[]
for folder in [O/'audit',O/'supplements/audit',O/'control_transition/audit']:
 for meta in sorted(folder.glob('*.metadata.json')):
  a=json.load(open(meta));tag=meta.name.replace('.metadata.json','');p=folder/f'{tag}.prompt.txt';assert hashlib.sha256(p.read_bytes()).hexdigest()==a['prompt_sha256'];assert not a['tools_used']
  for q in a['images']:assert hashlib.sha256((R/q['path']).read_bytes()).hexdigest()==q['sha256']
  checks.append({'call':str(meta.relative_to(O)),'model':a['model'],'prompt_verified':True,'image_count':len(a['images']),'tools_used':False})
for name,h in json.load(open(O/'frozen_references.json')).items():assert hashlib.sha256((O/'references'/name).read_bytes()).hexdigest()==h
for p in (O/'prefixes').glob('*.json'):
 c,t=p.stem.split('_');assert all(x['end']<=int(t) for x in json.load(open(p)))
(O/'validation.json').write_text(json.dumps({'calls_verified':len(checks),'checks':checks,'frozen_reference_hashes_unchanged':True,'causal_prefix_end_gates':True,'limits':'Modelreviews notgroundtruth; prompts forbidfightmemorybut image reviewers cansee broadcasttext; predictors onlytextpriorannotations, nofuturedata.'},indent=2));print('ANALYZED',len(checks),'auditcalls',len(summary),'stratifiedrows')
