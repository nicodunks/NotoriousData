from pathlib import Path
import cv2
from PIL import Image,ImageDraw
for path in Path('work/raw').glob('*.mp4'):
 cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
 times=list(range(15,int(dur),30)); sheet=Image.new('RGB',(960,200*((len(times)+3)//4)),(15,20,25)); draw=ImageDraw.Draw(sheet)
 for i,t in enumerate(times):
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,f=cap.read()
  if not ok:continue
  img=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)); img.thumbnail((240,170)); x=(i%4)*240;y=(i//4)*200
  sheet.paste(img,(x,y));draw.text((x+5,y+173),f'{t//60:02d}:{t%60:02d} ({t}s)',fill='white')
 out=Path('work')/(path.stem+'-contact.jpg');sheet.save(out);print(path,dur,out)
