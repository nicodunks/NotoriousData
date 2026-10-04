import cv2,json,subprocess,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg
R=Path.cwd();O=R/'outputs/experiment';refs=json.loads((O/'reviewed_labels.json').read_text());font='/System/Library/Fonts/Supplemental/Arial.ttf';F=lambda s:ImageFont.truetype(font,s);ids={'A':'01-standing','B':'02-takedown','C':'03-ground','D':'04-finish'};links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
for root in [91,112]:
 for f in range(5):
  a=root+1+4*f;links += [(root,a)]+[(j,j+1) for j in range(a,a+3)]
for a,b in [(23,40),(40,45),(45,50),(50,54),(54,59),(59,65),(65,71),(71,83),(83,91)]:links += [(j,j+1) for j in range(a,b-1)]
links += [(15,17),(17,18),(15,19),(16,20),(20,21),(16,22)]
for c in 'ABCD':
 pose=json.loads((R/'outputs/pose'/f'{ids[c]}__rtmw.json').read_text());cap=cv2.VideoCapture(str(O/'clips'/f'{c}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);ff=imageio_ffmpeg.get_ffmpeg_exe();p=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x850','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(O/f'rich-annotated-{c}.mp4')],stdin=subprocess.PIPE);i=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  t=i/fps;i+=1;label=refs[c][min(4,int(t//4))];im=Image.new('RGB',(960,850),'#111811');im.paste(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)),(0,0));d=ImageDraw.Draw(im);frame=pose['frames'][min(len(pose['frames'])-1,round(t*12))]
  for n,person in enumerate(frame['people']):
   k=person['keypoints'];col='#c6ee8b'
   for a,b in links:
    if k[a][2]>=.3 and k[b][2]>=.3 and all(0<=k[j][0]<960 and 0<=k[j][1]<540 for j in [a,b]):d.line((k[a][0],k[a][1],k[b][0],k[b][1]),fill=col,width=2 if a<23 else 1)
   for x,y,conf in k:
    if conf>=.3 and 0<=x<960 and 0<=y<540:d.ellipse((x-2,y-2,x+2,y+2),fill=col)
  d.rectangle((0,0,960,42),fill='#111811');d.text((16,10),f'CLIP {c}  {t:.2f}s | Sol reviewed events · RTMW whole-body estimates',font=F(19),fill='#c9f38e')
  d.text((18,556),f"Interval {label['start']}–{label['end']}s | end-state: {label['end_phase']} | visible contact outcomes",font=F(19),fill='#c9f38e')
  y=587
  gear=' | '.join(f'{k}: {v}' for k,v in label['fighters'].items())
  for line in textwrap.wrap(gear,105)[:2]:d.text((18,y),line,font=F(16),fill='#adbba9');y+=20
  active=[e for e in label['events'] if e['time_start']<=t<=e['time_end']]
  if not active:d.text((18,y+8),'No discrete event annotated at this instant.',font=F(19),fill='white')
  for e in active[:2]:
   text=f"{e['actor']} · {e['action']} Contact: {e['contact']} ({e['certainty']})."
   for line in textwrap.wrap(text,96)[:3]:d.text((18,y+5),line,font=F(17),fill='white');y+=21
  d.text((18,817),'Model-reviewed, no expert ground truth. RTMW landmark ownership is unverified.',font=F(16),fill='#adbba9')
  p.stdin.write(np.asarray(im).tobytes())
 p.stdin.close();assert p.wait()==0;cap.release();print('Rendered',c,i,'frames',flush=True)
