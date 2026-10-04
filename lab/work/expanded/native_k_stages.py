"""Clarify reversal initiation versus establishment, without changing prior labels."""
import cv2,math,json
from pathlib import Path
from PIL import Image,ImageDraw
import annotate as a
R=Path.cwd();O=R/'outputs/expanded/key_moments/native_review';a.O=O;p=O/'evidence/K_7_stages';p.mkdir(parents=True,exist_ok=True)
c=cv2.VideoCapture(str(R/'work/raw/PHzT9Wl8HYQ.mp4'));fps=c.get(5);times=[i/fps for i in range(math.floor(7.65*fps),math.ceil(8.85*fps))];images=[]
for page in range(0,len(times),2):
 sheet=Image.new('RGB',(1280,2216),'#111');dr=ImageDraw.Draw(sheet)
 for j,t in enumerate(times[page:page+2]):
  c.set(cv2.CAP_PROP_POS_MSEC,(1531+t)*1000);ok,f=c.read();assert ok;sheet.paste(Image.fromarray(cv2.cvtColor(f[:,320:1600],cv2.COLOR_BGR2RGB)),(0,j*1108));dr.text((8,j*1108+1086),f'K_7 t={t:.4f}s nativefullheightcrop x320:1600',fill='white')
 q=p/f'stages_{page//2:02d}.jpg';sheet.save(q,quality=80);images.append(q)
c.release();(p/'manifest.json').write_text(json.dumps(dict(source_offset=1531,requested_times=times,fps=fps,crop=[320,0,1600,1080],spatial_resampling=False,selection='Later criterion clarification of initiation versus established control. No prior answers/forecasts supplied to fresh model.'),indent=2))
context=sorted((O/'evidence/K_7').glob('sequence_*.jpg'))
prompt='''Fresh independent anonymous MMA visual review. Use only supplied timestamped full12Hz context(6.5–11) and selected native-rate full-height crops. No tools, externalfiles, internet, fighter names, memories, knownoutcomes, priorreviewanswers or predictions. Clothing-color IDs; identityunclear permitted. Explicitly distinguish A: initiation of a committed rolling/reversal attempt, B: first VISIBLY ESTABLISHED changed top/bottom CONTROL position, C: later consolidation. For major_control_transition, target onset is B, not A. A developing roll or brief head-height exchange is not by itself established changed control. Report whether first establishment canbe placed before8.000s, after8.000s, or unclear; never force exacttime when body occluded. Contact/impact is irrelevant to establishing positionalchange. Native requested times quantized within~1frame. Return JSONonly {case_id,fighters,roll_initiation:{occurrence,approximate_onset_or_interval,evidence_timestamps,visible_cues},control_establishment:{occurrence,approximate_onset_or_interval,evidence_timestamps,visible_top_bottom_cues,established_before_8:yes|no|unclear,uncertainties},consolidation:{evidence_timestamps,visible_cues},major_control_transition_for_7_to_11:{occurrence:yes|no|unclear,onset_or_interval,evidence_timestamps,limits},uncertainties}. There is no desired classification; source is retrospectively selected for criterion clarification, not population evaluation.'''
obj=a.call('K_7_stage_timing',prompt,context+images);obj['case_id']='K_7';obj['relation']='Independent stage-definition supplement; prior native and frozen reference outputs remain unchanged.';(O/'K_7_stage_timing.json').write_text(json.dumps(obj,indent=2));print('K7 STAGE TIMING COMPLETE',flush=True)
