"""Literal pixel crops for native selected windows, with full-frame sheets retained as context."""
import sys,cv2,json,math,concurrent.futures
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path.cwd()/'work/expanded'))
from annotate import R,O,S,call,RULES

def run(cid):
 a=O/'annotations'/f'{cid}.json'
 if not a.exists():raise RuntimeError(f'{cid}: initial adjudication must complete first')
 obj=json.loads(a.read_text());manifest=json.loads((O/'evidence'/cid/'dense_manifest.json').read_text());src=next(s for s in S if s[0]==cid)
 # Full-height center action strips avoid removing face/feet. Geometry is fixed and documented, never an identity claim.
 rect=(320,0,1600,1080) if cid in 'IJ' else ((160,0,1120,720) if cid in 'EF' else (80,0,1200,720))
 cap=cv2.VideoCapture(str(R/'work/raw'/f'{src[1]}.mp4'));fps=cap.get(5);images=[];records=[]
 for k,m in enumerate(manifest):
  frames=[]
  for idx in range(math.floor(m['start']*fps),math.ceil(m['end']*fps)):
   t=idx/fps;cap.set(cv2.CAP_PROP_POS_MSEC,(src[2]+t)*1000);ok,f=cap.read()
   if ok:frames.append((t,idx,f))
  for page in range(0,len(frames),2):
   group=frames[page:page+2];width=rect[2]-rect[0];height=rect[3]-rect[1];rowheight=height+28;sheet=Image.new('RGB',(width,rowheight*2),'#111');dr=ImageDraw.Draw(sheet)
   for j,(t,idx,f) in enumerate(group):
    crop=f[rect[1]:rect[3],rect[0]:rect[2]];sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(0,j*rowheight));dr.text((8,j*rowheight+height+6),f'{cid} t={t:.4f}s frame={idx} crop={rect}',fill='white')
   p=O/'evidence'/cid/f'action_crop_{k}_{page//2:02d}.jpg';sheet.save(p,quality=97);images.append(p)
  records.append(dict(window=k,start=m['start'],end=m['end'],rect_xyxy=rect,source_width=int(cap.get(3)),source_height=int(cap.get(4)),no_spatial_resampling=True,timestamps=[t for t,idx,f in frames],limitations='Fixed center strip can exclude peripheral hands/feet; original full frames retained. Broadcast image clarity is finite. Native file resolution is not a claim about model visual acuity.'))
 cap.release();(O/'evidence'/cid/'action_crop_manifest.json').write_text(json.dumps(records,indent=2))
 if not images:return
 detail=call(f'{cid}_action_crop_review',RULES+'\nReview this prior adjudication against literal native-resolution action crop sequence. These crops are full-height center strips and may exclude peripheral limbs; use accompanying full-frame context at each selected window. Original 2Hz discovery context is also supplied; contact between those sparse samples remains uncertain. Explicitly distinguish strike contact from sustained grip/body-to-body/control contact, and add contact_kind strike|grip|body_control|unclear to each contact claim. No invention of contact outside dense windows. Return {clip_id,window_reviews:[{start,end,observable_cues,evidence_timestamps,contact_claims:[{event_id,contact,evidence_timestamps,certainty}],visibility,uncertainties}],corrections:[{event_id,field,old,new,evidence_timestamps,reason}],remaining_disagreements,limits}. Do not force a resolution. Prior adjudication '+json.dumps(obj),sorted((O/'evidence'/cid).glob('discovery*.jpg'))+images+[O/'evidence'/cid/f'dense_{k}_00.jpg' for k in range(len(manifest))])
 (O/'annotations'/f'{cid}_detail.json').write_text(json.dumps(detail,indent=2));print(cid,'ACTION CROP REVIEW COMPLETE',len(images),flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,'EFGH'))
