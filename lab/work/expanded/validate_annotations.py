"""Mechanical expanded-label audit; does not measure accuracy."""
import json,hashlib,sys
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded';summary=[]
clip_ids=sys.argv[1] if len(sys.argv)>1 else 'EFGH'
for cid in clip_ids:
 p=O/'annotations'/f'{cid}.json'
 if not p.exists():continue
 obj=json.loads(p.read_text());audit=[]
 for stage in ['discovery','independent','adjudication','action_crop_review']:
  b=O/'audit'/f'{cid}_{stage}';meta=json.loads(b.with_suffix('.metadata.json').read_text());prompt=b.with_suffix('.prompt.txt').read_text()
  assert not meta['tools_used'];assert meta['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
  for im in meta['images']:
   assert hashlib.sha256((R/im['path']).read_bytes()).hexdigest()==im['sha256']
  audit.append(dict(stage=stage,model=meta['model'],seconds=meta['seconds'],images=len(meta['images'])))
 for e in obj.get('events',[]):
  assert e.get('contact') in ['evidence_supported_contact','evidence_supported_miss','unclear','not_applicable'],(cid,e)
  assert 0<=float(e.get('start',0))<=float(e.get('end',20))<=20,(cid,e)
 assert 'Alpha' not in json.dumps(obj) and 'Beta' not in json.dumps(obj)
 detail=json.loads((O/'annotations'/f'{cid}_detail.json').read_text())
 for w in detail.get('window_reviews',[]):
  for claim in w.get('contact_claims',[]):
   assert claim.get('contact') in ['evidence_supported_contact','evidence_supported_miss','unclear','not_applicable'],(cid,claim)
   assert claim.get('contact_kind') in ['strike','grip','body_control','unclear'],(cid,claim)
 summary.append(dict(detail_corrections=len(detail.get('corrections',[])),clip_id=cid,segments=len(obj.get('segments',[])),events=len(obj.get('events',[])),disagreements=len(obj.get('disagreements',[])),contact_counts={k:sum(e.get('contact')==k for e in obj.get('events',[])) for k in ['evidence_supported_contact','evidence_supported_miss','unclear','not_applicable']},audit=audit))
(O/('annotation_validation_extra.json' if clip_ids=='IJ' else 'annotation_validation.json')).write_text(json.dumps(dict(status='complete' if len(summary)==len(clip_ids) else 'partial',checks='JSON parses, taxonomy, event times, no old identifiers, model/tool provenance and exact image/prompt hashes; no gold accuracy measurement',clips=summary),indent=2));print(json.dumps(summary,indent=2))
