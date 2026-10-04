import hashlib,json
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/key_moments/native_review';out=[]
for tag in ['H_8_onset_supplement','K_7_stage_timing']:
 p=O/f'{tag}.json'
 if not p.exists():continue
 j=json.loads(p.read_text());b=O/'audit'/tag;m=json.loads(b.with_suffix('.metadata.json').read_text());assert not m['tools_used'];assert hashlib.sha256(b.with_suffix('.prompt.txt').read_bytes()).hexdigest()==m['prompt_sha256']
 for i in m['images']:assert hashlib.sha256((R/i['path']).read_bytes()).hexdigest()==i['sha256']
 if tag.startswith('H'):
  assert j['strike_attempt']['occurrence'] in ['yes','no','unclear']
  for lo,hi in j['strike_attempt'].get('onset_intervals',[]):assert 8<float(lo)<=float(hi)<12
 out.append(dict(tag=tag,model=m['model'],reasoning=m['reasoning'],seconds=m['seconds'],images=len(m['images']),tools_used=m['tools_used'],exact_prompt_and_image_hashes_verified=True))
(O/'supplement_validation.json').write_text(json.dumps(dict(relation='Later criterion/timing supplements only; frozen and previous native outputs unchanged.',lineages=out),indent=2));print(out)
