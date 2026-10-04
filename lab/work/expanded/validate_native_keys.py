"""Later reference comparison; never changes frozen labels or forecasts."""
import json,hashlib
from pathlib import Path
R=Path.cwd();B=R/'outputs/expanded/key_moments';O=B/'native_review';rows=[]
for key in ['H_4','H_8','H_12','H_16','G_8','E_16','K_7']:
 p=O/f'{key}.json'
 if not p.exists():continue
 obj=json.loads(p.read_text());cut=float(key.split('_')[1]);stages=['native_primary','native_independent','native_adjudication'];checks=[]
 for stage in stages:
  b=O/'audit'/f'{key}_{stage}';m=json.loads(b.with_suffix('.metadata.json').read_text());assert not m['tools_used'];assert hashlib.sha256(b.with_suffix('.prompt.txt').read_bytes()).hexdigest()==m['prompt_sha256']
  for im in m['images']:assert hashlib.sha256((R/im['path']).read_bytes()).hexdigest()==im['sha256']
  checks.append(dict(stage=stage,model=m['model'],images=len(m['images']),seconds=m['seconds']))
 refpath=(B/'control_transition'/'references'/f'{key}.json') if key.startswith('K_') else (B/'references'/f'{key}.json');ref=json.loads(refpath.read_text()) if refpath.exists() else None;targets=[]
 for t,n in obj.get('targets',{}).items():
  assert n.get('occurrence') in ['yes','no','unclear'],(key,t,n)
  # Explicitly list onset boundary problems rather than hiding them.
  boundary_flags=[v for v in n.get('onsets_seconds',[]) if not cut<float(v)<cut+4]
  original=ref.get('targets',{}).get(t,{}) if ref else {}
  targets.append(dict(target=t,native_occurrence=n.get('occurrence'),frozen_occurrence=original.get('occurrence'),same_occurrence=n.get('occurrence')==original.get('occurrence') if original else None,native_onsets=n.get('onsets_seconds',[]),frozen_onsets=original.get('onsets_seconds',[]),boundary_flags=boundary_flags))
 supplement_path=O/({'H_8':'H_8_onset_supplement.json','K_7':'K_7_stage_timing.json'}.get(key,'__none__'))
 supplement=json.loads(supplement_path.read_text()) if supplement_path.exists() else None
 rows.append(dict(timing_supplement=supplement,case_id=key,targets=targets,provenance=checks,reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest() if ref else None))
(O/'reference_comparison.json').write_text(json.dumps(dict(purpose='Separate later native review sensitivity evidence; frozen reference files and metrics are unchanged. Native models did not receive forecasts/reference answers.',status='complete' if len(rows)>=7 else 'partial',onset_semantics='K7 original native7.8333 is roll initiation; K7_stage_timing distinguishes established control after8.000/by8.475–8.500, matching frozen establishment convention. Do not use initiation to relabel K4. H8 supplement bounds a newly committed punch11.7784–11.8452 before12; contact remains unclear.',cases=rows),indent=2));print(json.dumps(rows,indent=2))
