import json,cv2,subprocess,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
R=Path.cwd();O=R/'outputs/expanded';(O/'color_videos').mkdir(exist_ok=True)
links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(17,18),(15,19),(16,20),(20,21),(16,22)]
for root in [91,112]:
 for f in range(5):
  a=root+1+f*4;links += [(root,a)]+[(j,j+1) for j in range(a,a+3)]
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19);checks=[]
for c in 'ABCD':
 ids=json.loads((O/'identity'/f'{c}-identities.json').read_text());poses=json.loads((R/'outputs/experiment/body_data'/f'{c}_rtmw_133.json').read_text());cap=cv2.VideoCapture(str(R/'outputs/experiment/clips'/f'{c}.mp4'));fps=cap.get(5)
 path=O/'color_videos'/f'{c}-colored-figures.mp4';p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x600','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE);i=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  t=i/fps;ix=min(len(poses['frames'])-1,round(t*poses['fps']));pose=poses['frames'][ix];ident=ids['frames'][min(len(ids['frames'])-1,ix)];im=Image.new('RGB',(960,600),'#10171c');im.paste(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)),(0,0));d=ImageDraw.Draw(im)
  for n,person in enumerate(pose['people']):
   match=next((z for z in ident['people'] if z['detection_index']==n),None);col=match['display_color'] if match else '#a1a1aa';k=person['keypoints']
   def valid(j):return len(k)>j and k[j][2]>=.3 and 0<=k[j][0]<960 and 0<=k[j][1]<540
   for a,b in links:
    if valid(a) and valid(b):d.line((k[a][0],k[a][1],k[b][0],k[b][1]),fill=col,width=2 if a<23 else 1)
   for j,(x,y,sc) in enumerate(k):
    if valid(j):rad=2 if j<23 else 1;d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=col)
  d.text((15,549),f'Clip {c} · {t:.2f}s · cyan / coral fighter estimates · gray unassigned',font=font,fill='white');d.text((15,575),'133-point RTMW estimates sampled at 12 Hz · joints and contact remain uncertain',font=font,fill='#a3b0b7');p.stdin.write(np.asarray(im).tobytes());i+=1
  if i==round(fps):im.save(O/'color_videos'/f'{c}-example.jpg',quality=96)
 p.stdin.close();assert p.wait()==0;cap.release();v=cv2.VideoCapture(str(path));ok,f=v.read();v.set(cv2.CAP_PROP_POS_FRAMES,max(0,i-1));last,_=v.read();checks.append(dict(clip=c,path=str(path.relative_to(O)),first_decoded=ok,last_decoded=last,frames=i,fps=fps,identity_counts=ids['counts']));v.release();print('RENDERED',c,i,flush=True)
(O/'color_videos/validation.json').write_text(json.dumps(checks,indent=2))
