import cv2,json,numpy as np,subprocess,imageio_ffmpeg
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/four_bout';(O/'clips').mkdir(exist_ok=True)
cases=[('2012','obgm6JNtyVo',250),('2013','BCOy-PG8EIw',223),('2018','khabib-mcgregor',675),('2021','6yu2AWK4rxo',596)]
for year,sid,start in cases:
 cap=cv2.VideoCapture(str(R/f'work/raw/{sid}.mp4'));cap.set(cv2.CAP_PROP_POS_MSEC,start*1000);fps=cap.get(cv2.CAP_PROP_FPS);ff=imageio_ffmpeg.get_ffmpeg_exe();out=O/'clips'/f'{year}-native-standing.mp4';proc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x590','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
 for j in range(round(8*fps)):
  ok,im=cap.read()
  if not ok:break
  canvas=np.full((590,960,3),(25,20,17),np.uint8);canvas[50:]=cv2.resize(im,(960,540));cv2.putText(canvas,f'{year} / source {start+j/fps:.2f}s / native evidence',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,235,220),1);cv2.putText(canvas,'Views and opponents differ; this is evidence, not an averaged style comparison.',(10,42),cv2.FONT_HERSHEY_SIMPLEX,.45,(195,195,195),1);proc.stdin.write(canvas.tobytes())
 proc.stdin.close();assert proc.wait()==0;print(year,flush=True)
