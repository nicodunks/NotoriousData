import cv2,math
from pathlib import Path
from PIL import Image,ImageDraw
R=Path.cwd();B=R/'work/experiment';O=R/'outputs/experiment'
for c in 'ABCD':
 p=B/'native'/c;p.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(O/'clips'/f'{c}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);bins={s:[] for s in range(20)};idx=0
 while True:
  ok,f=cap.read()
  if not ok:break
  t=idx/fps;idx+=1
  if t>=20:break
  f=f[:int(f.shape[0]*.88)];f=cv2.cvtColor(cv2.resize(f,(480,270)),cv2.COLOR_BGR2RGB);bins[int(t)].append((t,f))
 cap.release()
 for st,frames in bins.items():
  rows=math.ceil(len(frames)/6);sheet=Image.new('RGB',(2880,rows*290),'#111');draw=ImageDraw.Draw(sheet)
  for j,(t,f) in enumerate(frames):
   x=j%6*480;y=j//6*290;sheet.paste(Image.fromarray(f),(x,y));draw.text((x+6,y+272),f'{c} t={t:.4f}s',fill='white')
  sheet.save(p/f'{st:02d}.jpg',quality=96)
print('Native-rate sheets ready.')
