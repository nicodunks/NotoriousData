"""Recover completed reviews rejected only because transport fallback was misclassified as tool usage."""
import json,hashlib
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded'
for cid in 'EFGH':
 base=O/'audit'/f'{cid}_action_crop_review';dest=base.with_suffix('.json');log=base.with_suffix('.jsonl');pf=base.with_suffix('.prompt.txt')
 lines=[]
 for line in log.read_text().splitlines():
  try:lines.append(json.loads(line))
  except ValueError:pass
 tools=[x for x in lines if x.get('item',{}).get('type') not in [None,'reasoning','agent_message','error']]
 assert not tools,tools
 assert any(x.get('type')=='turn.completed' for x in lines),'Missing completed turn'
 obj=json.loads(dest.read_text().strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip());dest.write_text(json.dumps(obj,indent=2))
 m=json.loads((O/'evidence'/cid/'dense_manifest.json').read_text());images=sorted((O/'evidence'/cid).glob('discovery*.jpg'))
 for k in range(len(m)):images+=sorted((O/'evidence'/cid).glob(f'action_crop_{k}_*.jpg'))
 images += [O/'evidence'/cid/f'dense_{k}_00.jpg' for k in range(len(m))]
 meta=dict(model='gpt-6.1-sol',reasoning='high',seconds=None,tools_used=tools,transport_errors=[x for x in lines if x.get('item',{}).get('type')=='error'],prompt_sha256=hashlib.sha256(pf.read_bytes()).hexdigest(),images=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],usage=[x.get('usage') for x in lines if x.get('usage')],recovery='Exact saved prompt, completed JSON and transcript preserved; attached image list reproduced from deterministic action_crops.py ordering. Wall-clock duration unavailable because original tool guard rejected harmless transport fallback error.')
 base.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2));(O/'annotations'/f'{cid}_detail.json').write_text(json.dumps(obj,indent=2));print(cid,len(images),len(obj.get('corrections',[])),flush=True)
