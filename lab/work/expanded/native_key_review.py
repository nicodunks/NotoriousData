"""Independent future-only key-moment evidence revisit. Never changes frozen references or forecasts."""
import json,cv2,concurrent.futures,math,sys
from pathlib import Path
from PIL import Image,ImageDraw
import annotate as a
R=Path.cwd();ROOT=R/'outputs/expanded/key_moments';O=ROOT/'native_review';E=O/'evidence'
for p in [O,O/'audit',E]:p.mkdir(parents=True,exist_ok=True)
a.O=O
CASES=[('H',4),('H',8),('H',12),('H',16),('G',8)]
SOURCES={'H':('oliveira-poirier',550),'G':('khabib-mcgregor',200),'K':('PHzT9Wl8HYQ',1531),'E':('pereira-adesanya',100)}
RULES='''Review ONLY supplied anonymous timestamped future frames. No tools, files, internet, names, fight memory or known outcomes. No prediction or earlier reference answers are supplied. Clothing-color IDs only; unknown actor permitted. The target is a NEW commitment after cutoff, NOT every next movement. strike_attempt means newly committed punch/kick/knee/elbow including a ground strike, even if contact is unclear or missed. A clear reinitiation after reset counts. A single strike already underway at cutoff does not count. Feints, routine face/neck pressure, grip shifts or loose repositioning alone do not count. If trajectory cannot distinguish strike from grip/control, occurrence unclear; do not force yes. Contact is separate from attempted action. Takedown requires committed shot/trip/throw. Submission entry requires newly committed attack, not ordinary grip. Major control transition requires pass/reversal/escape or newly established dominant control, not microgrips or descent double-counting. All timing is requested-frame time, quantized within one native frame. Return JSON only. No accuracy/gold-truth claim.'''
SCHEMA='''Return {case_id,cutoff,horizon_end,fighters:[{clothing_id,description}],targets:{strike_attempt:{occurrence:yes|no|unclear,onsets_seconds:[],actors:[],evidence_timestamps:[],observable_commitment_and_reset_cues,ongoing_at_cutoff,contact_status:unclear|evidence_supported_contact|evidence_supported_miss|not_applicable,visibility_limits},takedown_attempt:{occurrence,onsets_seconds,actors,evidence_timestamps,evidence,visibility_limits},submission_entry:{occurrence,onsets_seconds,actors,evidence_timestamps,evidence,visibility_limits},major_control_transition:{occurrence,onsets_seconds,actors,evidence_timestamps,evidence,visibility_limits}},ambiguous_candidates:[{start,end,actor,action_candidates,strike_vs_control_cues,why_unclear}],zoom_requests:[{start,end,reason}],uncertainties:[]}. Occurrence yes requires a new onset in (cutoff,cutoff+4); preroll and endpoint are reference context only. Request <=3 native-rate 0.6s zoom windows around crucial ambiguous commitment/reset cues. Evidence timestamps required.'''

def getframe(cap,off,t):
 cap.set(cv2.CAP_PROP_POS_MSEC,(off+t)*1000);ok,f=cap.read();assert ok;return f

def prepare(cid,cut):
 key=f'{cid}_{cut}';src,off=SOURCES[cid];cap=cv2.VideoCapture(str(R/'work/raw'/f'{src}.mp4'));fps=cap.get(5);p=E/key;p.mkdir(exist_ok=True);seq=[]
 times=[cut-.5+i/12 for i in range(55)] # endpoint included only as reference context
 for t in times:seq.append((t,getframe(cap,off,t)))
 images=[]
 for page in range(0,len(seq),4):
  g=seq[page:page+4];h,w=g[0][1].shape[:2];sheet=Image.new('RGB',(w*2,(h+28)*2),'#111');dr=ImageDraw.Draw(sheet)
  for j,(t,f) in enumerate(g):
   x=j%2*w;y=j//2*(h+28);sheet.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(x,y));dr.text((x+8,y+h+6),f'{key} t={t:.4f}s cutoff={cut} full native frame',fill='white')
  q=p/f'sequence_{page//4:02d}.jpg';sheet.save(q,quality=72 if cid=='K' else 82);images.append(q)
 cap.release();(p/'manifest.json').write_text(json.dumps(dict(case_id=key,cutoff=cut,horizon_end=cut+4,sampling_hz=12,source_path=str(R/'work/raw'/f'{src}.mp4'),source_offset=off,native_fps=fps,requested_relative_timestamps=times,crop=None,spatial_resampling=False,preroll_reference_only=[cut-.5,cut],endpoint_reference_only=cut+4,limits='Timestamp seeks quantized within native frame. Camera cropping/occlusion/compression remain. Anonymous to model except visual broadcast content.'),indent=2));return images

def review(case):
 cid,cut=case;key=f'{cid}_{cut}';images=prepare(cid,cut);p=E/key
 primary=a.call(f'{key}_native_primary',RULES+f'\nCase {key}; cutoff={cut}, future ends={cut+4}.\n'+SCHEMA,images);print(key,'primary',flush=True)
 independent=a.call(f'{key}_native_independent',RULES+f'\nFresh independent review. Case {key}; cutoff={cut}, future ends={cut+4}.\n'+SCHEMA,images);print(key,'independent',flush=True)
 windows=[]
 for r in primary.get('zoom_requests',[])+independent.get('zoom_requests',[]):
  st=max(cut-.5,min(cut+3.4,float(r.get('start',cut))));en=st+.6
  if all(abs(st-x)>0.4 for x,y in windows):windows.append((st,en))
  if len(windows)>=3:break
 src,off=SOURCES[cid];cap=cv2.VideoCapture(str(R/'work/raw'/f'{src}.mp4'));fps=cap.get(5);zoom=[];manifest=[]
 for k,(st,en) in enumerate(windows):
  ts=[i/fps for i in range(math.floor(st*fps),math.ceil(en*fps))]
  for page in range(0,len(ts),2):
   group=ts[page:page+2];rect=(320,0,1600,1080) if cid=='K' else (80,0,1200,720);width=rect[2]-rect[0];height=rect[3]-rect[1];rowheight=height+28;sheet=Image.new('RGB',(width,rowheight*2),'#111');dr=ImageDraw.Draw(sheet)
   for j,t in enumerate(group):
    f=getframe(cap,off,t);crop=f[rect[1]:rect[3],rect[0]:rect[2]];sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(0,j*rowheight));dr.text((8,j*rowheight+height+6),f'{key} t={t:.4f}s native crop {rect} fullheight',fill='white')
   q=p/f'zoom_{k}_{page//2:02d}.jpg';sheet.save(q,quality=80 if cid=='K' else 86);zoom.append(q)
  manifest.append(dict(start=st,end=en,requested_timestamps=ts,crop_xyxy=list(rect),spatial_resampling=False))
 cap.release();(p/'zoom_manifest.json').write_text(json.dumps(manifest,indent=2))
 final=a.call(f'{key}_native_adjudication',RULES+f'\nCase {key}, cutoff={cut}, future ends={cut+4}. Compare independent reviews with full 12Hz context and selected native-rate crops. Neither review is authoritative. Preserve strike-vs-control uncertainty. Return schema targets as in primary plus disagreements:[{{claim,primary,independent,native_cues,evidence_timestamps,resolution,remaining_uncertainty}}], review_limits. No frozen reference answers or forecasts are available. PRIMARY '+json.dumps(primary)+'\nINDEPENDENT '+json.dumps(independent),images+zoom)
 final['case_id']=key;final['reference_relation']='Separate later native review; frozen reference untouched. Native-review models did not receive reference answers or predictions.';(O/f'{key}.json').write_text(json.dumps(final,indent=2));print(key,'COMPLETE',flush=True)
if __name__=='__main__':
 if '--K' in sys.argv:CASES=[('K',7)]
 if '--E16' in sys.argv:CASES=[('E',16)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(review,CASES))
