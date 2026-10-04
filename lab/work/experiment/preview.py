import cv2,json,subprocess,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg
from colors import colorize
R=Path.cwd();O=R/'outputs/experiment';O.mkdir(exist_ok=True);font='/System/Library/Fonts/Supplemental/Arial.ttf';F=lambda s:ImageFont.truetype(font,s)
labels=colorize(json.loads((O/'labels/A.json').read_text())[0]);pose=json.loads((R/'outputs/pose/01-standing__rtmw.json').read_text());cap=cv2.VideoCapture(str(O/'clips/A.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
ff=imageio_ffmpeg.get_ffmpeg_exe();p=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x810','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(O/'annotation-preview.mp4')],stdin=subprocess.PIPE)
links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(17,18),(15,19),(16,20),(20,21),(16,22)]
for root in [91,112]:
 for f in range(5):
  start=root+1+4*f;links += [(root,start)]+[(a,a+1) for a in range(start,start+3)]
for start,end in [(23,40),(40,45),(45,50),(50,54),(54,59),(59,65),(65,71),(71,83),(83,91)]:links += [(a,a+1) for a in range(start,end-1)]
for i in range(round(4*fps)):
 ok,fr=cap.read();assert ok;t=i/fps;im=Image.new('RGB',(960,810),'#111811');im.paste(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)),(0,0));d=ImageDraw.Draw(im)
 # Existing 12-Hz skeleton layer shown explicitly as an unchanged estimate, not Sol coordinates.
 frame=min(pose['frames'],key=lambda x:abs(x['time']-t))
 for n,person in enumerate(frame['people']):
  k=person['keypoints'];hipx=(k[11][0]+k[12][0])/2;col='#7de47c' if hipx<480 else '#f5df69'
  for a,b in links:
   if k[a][2]>=.3 and k[b][2]>=.3 and all(0<=k[j][0]<960 and 0<=k[j][1]<540 for j in [a,b]):d.line((k[a][0],k[a][1],k[b][0],k[b][1]),fill=col,width=2 if a<23 else 1)
  for j,(x,y,score) in enumerate(k):
   if score>=.3 and 0<=x<960 and 0<=y<540:
    r=2 if j<23 else 1;d.ellipse((x-r,y-r,x+r,y+r),fill=col)
 d.rectangle((0,0,960,42),fill='#111811');d.text((16,10),f'CLIP A  {t:.2f}s  |  RTMW face/hands/feet estimate + Sol event layer',font=F(19),fill='#c9f38e')
 d.text((18,558),'FIRST-PASS LABELS — DENSE REVIEW PENDING',font=F(17),fill='#c9f38e')
 d.text((18,586),'Green shorts   |   Yellow shorts   |   phase: standing',font=F(18),fill='white')
 active=[e for e in labels['events'] if e['time_start']<=t<=e['time_end']]
 y=618
 for e in active[:2]:
  text=f"{e['actor']} · {e['action']} Contact: {e['contact']}."
  for line in textwrap.wrap(text,95):d.text((18,y),line,font=F(18),fill='white');y+=23
 d.text((18,775),'Contact windows are candidates. No confirmed impact point or force is claimed.',font=F(16),fill='#aebbad')
 if i==30:im.save(O/'annotation-example.jpg',quality=95)
 p.stdin.write(np.asarray(im).tobytes())
p.stdin.close();assert p.wait()==0;cap.release()
# Three original frames provide an evidence strip without inferred impact markers.
strip=Image.new('RGB',(1920,450),'#111811');d=ImageDraw.Draw(strip);cap=cv2.VideoCapture(str(O/'clips/A.mp4'))
for j,t in enumerate([.875,1.125,1.375]):
 cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,fr=cap.read();assert ok;strip.paste(Image.fromarray(cv2.cvtColor(cv2.resize(fr,(640,360)),cv2.COLOR_BGR2RGB)),(j*640,38));d.text((j*640+14,10),f't={t:.3f}s',font=F(20),fill='#c9f38e')
d.text((14,411),'Yellow kick attempt: extension and retraction visible; clean contact remains unverified.',font=F(23),fill='white');strip.save(O/'contact-evidence-example.jpg',quality=95)
cap.release();(O/'annotation-example.json').write_text(json.dumps(labels,indent=2));print('Preview ready.')
