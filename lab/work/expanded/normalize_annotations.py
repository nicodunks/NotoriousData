"""Normalize stable clip join keys while retaining exact model results in audit/."""
import json
from pathlib import Path
O=Path('outputs/expanded')
for cid in 'EFGHIJ':
 for suffix in ['', '_detail']:
  p=O/'annotations'/f'{cid}{suffix}.json'
  if not p.exists():continue
  o=json.loads(p.read_text())
  if o.get('clip_id')!=cid:o['model_returned_clip_id']=o.get('clip_id');o['clip_id']=cid
  if cid in 'IJ':o['annotation_causality']='Retrospective full20-second image review; not a strictly causal prefix label or forecast input.'
  o['annotation_scope']='Exploratory model evidence annotation; contacts include grips and body control, not necessarily landed strikes. Selected dense windows do not verify all attempted contacts.'
  p.write_text(json.dumps(o,indent=2))
