import json,hashlib
from pathlib import Path
p=Path('outputs/expanded/annotation_manifest.json');ms=json.loads(p.read_text())
for m in ms:
 m['overlap_warning']='F overlaps root S04 source interval; treat as same source evidence, not an independent sample' if m['id']=='F' else 'All E-H originate from the same three fights as A-D; intervals differ, fights are not independent.'
 m['sampling_limits']=['2Hz discovery can miss contact and brief actions between samples.','Native-rate selected windows explore ambiguities, not every attempted contact.','Full frames preserve source content; broadcast camera crops and occlusion remain.','Images contain broadcast names; prompt excludes names and fight memory, but this is not a blinded identity test.','Model frontend may resize images; native file pixels do not prove native model acuity.','Frame reads use timestamp seeking; requested timestamps may be quantized to the nearest native decoded frame (~33ms). No millisecond-level impact-time accuracy claimed.']
 m['clip_sha256']=hashlib.sha256(Path(m['clip']).read_bytes()).hexdigest();m['discovery_sha256']={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in m['discovery_sheets']}
p.write_text(json.dumps(ms,indent=2))
