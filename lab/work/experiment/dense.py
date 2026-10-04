import cv2,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path.cwd();B=ROOT/'work/experiment';ss=json.loads((ROOT/'work/samples.json').read_text())
for name,s in zip('ABCD',ss):
 p=B/'dense'/name;p.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(ROOT/'work/raw'/f'{s["source"]}.mp4'))
 for st in range(20):
  sheet=Image.new('RGB',(1920,1160),'#111');draw=ImageDraw.Draw(sheet)
  for j in range(16):
   t=st+j/16;cap.set(cv2.CAP_PROP_POS_MSEC,(s['start']+t)*1000);ok,f=cap.read();assert ok
   f=f[:int(f.shape[0]*.88)];f=cv2.cvtColor(cv2.resize(f,(480,270)),cv2.COLOR_BGR2RGB);x=j%4*480;y=j//4*290;sheet.paste(Image.fromarray(f),(x,y));draw.text((x+5,y+272),f'{name} t={t:.4f}s',fill='white')
  sheet.save(p/f'{st:02d}.jpg',quality=95)
 cap.release()
print('Dense independent-review sheets ready, 16 Hz.')
