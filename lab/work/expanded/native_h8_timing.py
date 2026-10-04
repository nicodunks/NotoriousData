"""Independent late-horizon onset precision supplement; prior references/reviews remain untouched."""
import cv2,json,math
from pathlib import Path
from PIL import Image,ImageDraw
import annotate as a
from native_key_review import RULES
R=Path.cwd();O=R/'outputs/expanded/key_moments/native_review';a.O=O;p=O/'evidence/H_8_timing';p.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(R/'work/raw/oliveira-poirier.mp4'));fps=cap.get(5);times=[i/fps for i in range(math.floor(11.5*fps),math.ceil(12*fps))];images=[]
for page in range(0,len(times),2):
 sheet=Image.new('RGB',(1120,1496),'#111');dr=ImageDraw.Draw(sheet)
 for j,t in enumerate(times[page:page+2]):
  cap.set(cv2.CAP_PROP_POS_MSEC,(550+t)*1000);ok,f=cap.read();assert ok;sheet.paste(Image.fromarray(cv2.cvtColor(f[:,80:1200],cv2.COLOR_BGR2RGB)),(0,j*748));dr.text((8,j*748+726),f'H_8 t={t:.4f}s fullheight native crop x80:1200',fill='white')
 q=p/f'late_{page//2:02d}.jpg';sheet.save(q,quality=86);images.append(q)
cap.release();(p/'manifest.json').write_text(json.dumps(dict(cutoff=8,horizon_end=12,requested_timestamps=times,crop=[80,0,1200,720],fps=fps,source_offset=550,selection='Retrospective supplementary review to close late-horizon native-frame coverage gap; no forecasts or prior answers supplied to fresh model.'),indent=2))
context=sorted((O/'evidence/H_8').glob('sequence_*.jpg'))
prompt=RULES+'\nCaseH_8: future(8,12), endpoint12 referenceonly. Full12Hz context plus last0.5s native crops. Fresh independent review: identify any NEW strike commitment, including ground strikes, before12; distinguish a newwindup from release/adjustment of hand/head control. Do not require observable impact to identify attempted strike. Do not assign intent from proximity alone. No priorreview or forecasts supplied. Return {case_id,strike_attempt:{occurrence:yes|no|unclear,actors:[],onset_seconds:[],onset_intervals:[],commitment_and_reset_cues,evidence_timestamps:[],contact_status,visibility_limits},late_candidates:[{start,end,actor,observable_motion,action_candidates,why_distinguishable_or_unclear}],limits:[]}. Use bounded approximate onset interval when exactframe cannotbe established; never invent a precise time.'
obj=a.call('H_8_onset_supplement',prompt,context+images);obj['case_id']='H_8';obj['relation']='Independent later timing supplement; prior native review and frozen references unchanged.';(O/'H_8_onset_supplement.json').write_text(json.dumps(obj,indent=2));print('H8 TIMING SUPPLEMENT COMPLETE',flush=True)
