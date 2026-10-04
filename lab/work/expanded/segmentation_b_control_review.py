import sys,json,cv2,hashlib,numpy as np,concurrent.futures
from PIL import Image,ImageDraw
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as base
O=R/'outputs/expanded/key_moments/native_review';O.mkdir(exist_ok=True);(O/'audit').mkdir(exist_ok=True);(O/'evidence').mkdir(exist_ok=True);base.O=O
source=R/'outputs/experiment/clips/B.mp4';man={}
def evidence(t):
 cap=cv2.VideoCapture(str(source));fps=cap.get(5);n=int(cap.get(7));requests=[t-.5,t-.25]+[t+j/12 for j in range(48)];samples=[];images=[]
 for j,when in enumerate(requests):
  ix=int(np.floor(when*fps+1e-8));ix=max(0,min(ix,n-1));cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,im=cap.read();assert ok
  samples.append({'native_frame_index':ix,'actual_time_seconds':ix/fps,'requested_time_seconds':when,'scope':'preroll' if j<2 else 'target'})
  if j%4==0:sheet=Image.new('RGB',(1920,1136),'#10171c');draw=ImageDraw.Draw(sheet)
  x=(j%4)%2*960;y=(j%4)//2*568;sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+8,y+546),f'Anonymous sequence B actual t={ix/fps:.5f}s '+('reference-only preroll' if j<2 else 'target horizon'),fill='white')
  if j%4==3 or j==len(requests)-1:
   p=O/'evidence'/f'B_{t}_{j//4:02d}.jpg';sheet.save(p,quality=97);images.append(p)
 cap.release();assert all(s['actual_time_seconds']<t+4 for s in samples);man[str(t)]={'cutoff':t,'horizon':4,'sampling_hz':12,'fps':fps,'samples':samples,'images':[str(p.relative_to(R)) for p in images]};return images
PROMPT='''Use ONLY the supplied ordered timestamped anonymous images. No tools, web, files, names, fight memory, external knowledge or future outcome inference. Review major_control_transition occurrences in the specified half-open target horizon. This is a fresh independent visual reference; no predictions or earlier label decisions are supplied. The images are native decoded960x540 frames sampled approximately12Hz; actual source timestamps are printed. Reference-only preroll clarifies actions already ongoing, and is never itself scored.
Definition: newly visibly established meaningful pass, reversal, positional escape or dominant control. Exclude small grips, base adjustments, brief screen-position changes, mere position above another body, and the descent belonging to an ongoing takedown. A distinct new established control after engagement may count if visible evidence establishes it. A control need not persist to the end, but mere fleeting overlap is insufficient. Do not infer unseen grips, hidden hooks, submission, or success from a plausible posture. Clothing-color IDs only; no Alpha/Beta or fixedscreen-side IDs. If visibility leaves meaningful control or its new onset unresolved, return unclear rather than manufacture yes/no. A no label requires enough observable coverage, otherwise unclear. No contact/damage/outcome claims.
Return JSON {cutoff,horizon,end_exclusive,major_control_transition:{occurrence:yes|no|unclear,events:[{onset_interval_seconds:[start,end],first_visible_seconds,actor_clothing,new_control_observed,prior_control_observed,evidence:[{time_seconds,observation}],certainty,limits}],evidence_for_negative_or_unclear:[],visibility_limits:[]},overall,limitations:[]}. All onset/evidence times actualsourceclipseconds. Return JSON only.'''
def run(t):
 a=base.call(f'B_control_{t}_{t+4}',PROMPT+f'\nTarget horizon [{t},{t+4}) seconds.',evidence(t));assert a['major_control_transition']['occurrence'] in ['yes','no','unclear'];(O/f'B_control_{t}_{t+4}.json').write_text(json.dumps(a,indent=2));print('CONTROL REVIEWED',t,flush=True);return a
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:reviews=list(ex.map(run,[6,7]))
 (O/'B_control.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','independent':True,'tool_free':True,'native_sampling_hz':12,'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target_definition':'new visibly established meaningful pass/reversal/escape/dominantcontrol; exclude microgrips, merepositionabove, same takedowndescent','sampling':man,'reviews':reviews,'not_gold_truth':True},indent=2));print('COMPLETE',flush=True)
