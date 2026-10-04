import sys,json,cv2,numpy as np,concurrent.futures
from pathlib import Path
R=Path.cwd();W=R/'work/expanded';sys.path.insert(0,str(W));import causal_v2 as base
O=R/'outputs/expanded/segmentation/review';O.mkdir(exist_ok=True);(O/'audit').mkdir(exist_ok=True);base.O=O
D=json.load(open(R/'outputs/expanded/segmentation/sam2-rtmw-c-d-0-4.json'))
COL={'fighter_black':(238,211,34),'fighter_red':(133,113,251)}
E=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for root in (91,112):
 for j in range(5):
  chain=[root]+list(range(root+1+j*4,root+5+j*4));E+=list(zip(chain,chain[1:]))
for a,z in [(23,40),(40,45),(45,49),(49,55),(55,59),(59,64),(64,68),(68,76),(76,82),(82,91)]: E+=list(zip(range(a,z-1),range(a+1,z)))
inputs={c:[] for c in 'CD'}
for row in D['frames']:
 c=row['clip'];fr=row['sample_index_12hz'];stem=f'{c}-{fr:03d}';orig=R/f'outputs/expanded/segmentation/{stem}-original.png';im=cv2.imread(str(orig));panels=[im.copy()];paths=[]
 for cond,kname in [('baseline','baseline_keypoints'),('opponent_masked','opponent_masked_keypoints')]:
  pic=im.copy()
  for key,person in row['fighters'].items():
   col=COL[key];kp=np.array(person[kname]);valid=(kp[:,0]>=0)&(kp[:,0]<960)&(kp[:,1]>=0)&(kp[:,1]<540)&(kp[:,2]>=2)
   for a,b in E:
    if valid[a] and valid[b]:cv2.line(pic,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),col,1,cv2.LINE_AA)
   for j in np.where(valid)[0]:cv2.circle(pic,tuple(kp[j,:2].astype(int)),2,col,-1,cv2.LINE_AA)
   b=np.array(person['box']);cv2.rectangle(pic,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),col,1)
  cv2.rectangle(pic,(0,0),(960,45),(16,23,28),-1);cv2.putText(pic,f'{c} t={row["time_seconds"]:.3f}s {cond}: Black=Cyan Red=Coral; candidates only',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.53,(240,240,240),1,cv2.LINE_AA)
  p=O/f'{stem}-{cond}-v2.png';cv2.imwrite(str(p),pic);paths.append(p);panels.append(pic)
 maskinputs=[]
 for target in ('fighter_black','fighter_red'):
  opp='fighter_red' if target=='fighter_black' else 'fighter_black';mask=cv2.imread(str(R/row['fighters'][opp]['mask_path']),0)>0;pic=im.copy();pic[mask]=np.median(im.reshape(-1,3),axis=0).astype(np.uint8)
  cv2.rectangle(pic,(0,0),(960,40),(16,23,28),-1);cv2.putText(pic,f'{c} t={row["time_seconds"]:.3f}s {target}: opponent mask removed input',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  p=O/f'{stem}-{target}-masked-input.png';cv2.imwrite(str(p),pic);maskinputs.append(p)
 p=O/f'{stem}-comparison-v2.png';cv2.imwrite(str(p),np.hstack(panels))
 inputs[c]+=[orig,p]+paths+maskinputs
prompt='''Use ONLY the supplied images. No tools, web, files, known fighter names, fight memory, or hidden-joint inference. This is an independent visual ownership review of two pretrained RTMW candidate conditions: baseline original image, and opponent-mask-removed diagnostic inference. All skeletons are rendered on ORIGINAL pixels for comparison; additional diagnostic input images show what the pose model actually saw after removal. Cyan #22d3ee identifies the Black/gray-shorts candidate (light hair); Coral #fb7185 identifies Red-shorts candidate. These display colors do not establish correct ownership. Inspect original visible anatomy first; plot line position, mask selection and high scores are NOT truth. The mask can erase own visible limbs when fighters overlap, or invent an occluded limb. State such risks. Do not claim numeric accuracy or detailed finger fidelity unless visibly supported. A hand near a head need not be that head owner's hand. Frame labels 000 and048 are 12Hz sample indices; actual times are0 and4seconds, not nativeframe48. Preprocessing originals are960x540; no claim full broadcast resolution. Judge support against this evidence only.
For EACH time0,4; EACH fighterBlack/gray,Red; EACH conditionbaseline,opponent_masked; assign limbgroup hands/feet/elbows/knees/head/torso status supported|mixed|unclear with concise visible evidence. supported means the shown visible candidate locations align with that fighter's own anatomy, not the opponent; mixed means at least clearly visible incorrect ownership/location or combination of correct and misplaced/hidden anatomy; unclear means visibility cannot resolve. Anatomical left/right may remain unknown. For each fighter/time compare conditions: improves|harms|neutral|unclear, naming exactly which visibly supported parts changed and any original own-body pixels erased. If no improvement is established say so; unknown is useful. Two frames perclip only: do not infer temporal trajectories, contact, damage, or populationperformance.
Return JSON {clip,assessments:[{time_seconds,fighter_clothing,condition,groups:{hands:{status,evidence},feet:{status,evidence},elbows:{status,evidence},knees:{status,evidence},head:{status,evidence},torso:{status,evidence}}}],comparisons:[{time_seconds,fighter_clothing,effect,visible_improvements:[],visible_harms:[],unknowns:[],mask_erasure_evidence}],overall,limitations:[]}. Exactly8 assessments and4 comparisons perclip. Return valid JSON only.'''
def run(c):
 a=base.call(f'{c}_independent_sol_ownership',prompt+'\nReviewclip '+c,inputs[c]);assert len(a['assessments'])==8 and len(a['comparisons'])==4
 for r in a['assessments']:
  assert r['condition'] in ['baseline','opponent_masked'];assert set(r['groups'])==set(['hands','feet','elbows','knees','head','torso']);assert all(g['status'] in ['supported','mixed','unclear'] for g in r['groups'].values())
 (O/f'{c}-review.json').write_text(json.dumps(a,indent=2));print('REVIEWED',c,flush=True);return a
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:allreviews=list(ex.map(run,'CD'))
 (O/'combined-review.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','no_tools':True,'new_inference':False,'review_scope':'4 selected frames; visible candidate ownership, not jointaccuracy','reviews':allreviews},indent=2))
 print('COMPLETE',flush=True)
