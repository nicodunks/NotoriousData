import cv2,json,hashlib,shutil
from pathlib import Path
O=Path('outputs/experiment');checks=[]
for p in sorted(O.glob('rich-annotated-*.mp4')):
 c=cv2.VideoCapture(str(p));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);ok1,_=c.read();c.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok2,fr=c.read();assert ok1 and ok2 and n>=590
 if p.name=='rich-annotated-A.mp4':
  c.set(cv2.CAP_PROP_POS_MSEC,1100);ok,f=c.read();assert ok;cv2.imwrite(str(O/'reviewed-annotation-example.jpg'),f)
 checks.append(dict(file=p.name,frames=n,fps=fps,width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),first_last_decode=True));c.release()
for d in ['frames','dense','native']:
 files=[]
 for p in (Path('work/experiment')/d).glob('*/*.jpg'):
  target=O/'frame_evidence'/d/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy(p,target)
  files.append(dict(original_path=str(p),archive_path=str(target.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 checks.append(dict(input_set=d,files=files))
(O/'artifact_checks.json').write_text(json.dumps(checks,indent=2));print('All videos decoded; 200 model-input sheets archived; reviewed still saved.')
