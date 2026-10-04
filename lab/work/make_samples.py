import json,subprocess
from pathlib import Path
import imageio_ffmpeg
root=Path.cwd(); out=root/'outputs';out.mkdir(exist_ok=True);(root/'work/clips').mkdir(exist_ok=True)
samples=[
 dict(id='01-standing',title='Pereira–Adesanya: standing exchanges',source='pereira-adesanya',start=250,duration=20,url='https://www.youtube.com/watch?v=9gLe0JfKz8w',channel='UFC Eurasia',focus='Striking motion, feet, hands, referee rejection'),
 dict(id='02-takedown',title='Khabib–McGregor: entry into wrestling',source='khabib-mcgregor',start=20,duration=20,url='https://www.youtube.com/watch?v=JuBBIJ7adjM',channel='UFC',focus='Standing-to-ground transition, overlapping fighters'),
 dict(id='03-ground',title='Oliveira–Poirier: ground control',source='oliveira-poirier',start=400,duration=20,url='https://www.youtube.com/watch?v=clG3LV28bC0',channel='ufcespanol',focus='Ground occlusion, ownership of limbs, top/bottom'),
 dict(id='04-finish',title='Oliveira–Poirier: back-control finish',source='oliveira-poirier',start=675,duration=20,url='https://www.youtube.com/watch?v=clG3LV28bC0',channel='ufcespanol',focus='Back control, submission sequence, phase boundaries')]
ff=imageio_ffmpeg.get_ffmpeg_exe()
for s in samples:
 p=root/'work/clips'/f'{s["id"]}.mp4';s['path']=str(p)
 subprocess.run([ff,'-v','error','-y','-ss',str(s['start']),'-i',str(root/'work/raw'/f'{s["source"]}.mp4'),'-t',str(s['duration']),'-vf','fps=12,scale=960:-2','-an','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p',str(p)],check=True)
 print(p,flush=True)
(root/'work/samples.json').write_text(json.dumps(samples,indent=2))
