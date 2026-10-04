"""Reduce JPEG transport bytes without changing native pixel dimensions. High-quality originals retained."""
from pathlib import Path
from PIL import Image
import hashlib,json
O=Path('outputs/expanded');records=[]
for cid in 'IJ':
 for p in sorted((O/'evidence'/cid).glob('discovery*.jpg')):
  dest=p.with_name('model_'+p.name)
  with Image.open(p) as im:im.save(dest,quality=72,optimize=True);dims=im.size
  records.append(dict(clip_id=cid,original=str(p),model_input=str(dest),pixel_dimensions=dims,spatial_resampling=False,jpeg_quality=72,original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_input_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),limit='Additional JPEG compression may weaken fine cues; native-rate action review is needed for ambiguous contact.'))
(O/'input_compression_manifest.json').write_text(json.dumps(records,indent=2))
