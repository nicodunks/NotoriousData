import cv2,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
ROOT=Path.cwd(); BASE=ROOT/'work/experiment'; OUT=ROOT/'outputs/experiment'
(BASE/'frames').mkdir(exist_ok=True); (OUT/'clips').mkdir(exist_ok=True)
samples=json.loads((ROOT/'work/samples.json').read_text()); manifest=[]
for name,s in zip('ABCD',samples):
 raw=ROOT/'work/raw'/f'{s["source"]}.mp4'; cap=cv2.VideoCapture(str(raw)); fps=cap.get(cv2.CAP_PROP_FPS)
 p=BASE/'frames'/name;p.mkdir(exist_ok=True)
 for block in range(10):
  sheet=Image.new('RGB',(1920,1160),'#111111'); draw=ImageDraw.Draw(sheet)
  for j in range(16):
   t=block*2+j/8;cap.set(cv2.CAP_PROP_POS_MSEC,(s['start']+t)*1000);ok,fr=cap.read()
   if not ok:raise RuntimeError((name,t))
   # Remove scoreboard/name band. Preserve the cage action; exact crop documented.
   fr=fr[:int(fr.shape[0]*.88),:];fr=cv2.cvtColor(cv2.resize(fr,(480,270)),cv2.COLOR_BGR2RGB)
   x=(j%4)*480;y=(j//4)*290;sheet.paste(Image.fromarray(fr),(x,y));draw.text((x+6,y+273),f'{name}  t={t:.3f}s',fill='white')
  sheet.save(p/f'{block*2:02d}-{block*2+2:02d}.jpg',quality=94)
 cap.release()
 clip=OUT/'clips'/f'{name}.mp4'
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss',str(s['start']),'-i',str(raw),'-t','20','-vf','scale=960:-2','-an','-c:v','libx264','-crf','18',str(clip)],check=True)
 manifest.append(dict(anonymous_id=name,source_sample=s['id'],source_offset=s['start'],source_fps=fps,source_url=s['url'],duration=20,frame_sampling_hz=8,scoreboard_crop='bottom 12% removed for model inputs',sheets=[dict(path=str(q.relative_to(ROOT)),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(p.glob('*.jpg'))]))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Prepared 40 chronological sheets and 4 original-FPS review clips.')
