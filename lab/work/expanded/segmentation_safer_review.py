import sys,json,cv2,numpy as np,hashlib,concurrent.futures
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as base
O=R/'outputs/expanded/segmentation/review/safer';O.mkdir(exist_ok=True);(O/'audit').mkdir(exist_ok=True);base.O=O
old=json.load(open(R/'outputs/expanded/segmentation/sam2-rtmw-c-d-0-4.json'));new=json.load(open(R/'outputs/expanded/segmentation/safer_variants/results.json'))
COL={'fighter_black':(238,211,34),'fighter_red':(133,113,251)};E=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for root in (91,112):
 for j in range(5):
  chain=[root]+list(range(root+1+j*4,root+5+j*4));E+=list(zip(chain,chain[1:]))
for a,z in [(23,40),(40,45),(45,49),(49,55),(55,59),(59,64),(64,68),(68,76),(76,82),(82,91)]:E+=list(zip(range(a,z-1),range(a+1,z)))
inputs={};maps={}
for c in 'CD':
 row=next(r for r in old['frames'] if r['clip']==c and r['sample_index_12hz']==48);orig=R/f'outputs/expanded/segmentation/{c}-048-original.png';im=cv2.imread(str(orig));inputs[c]=[orig];maps[c]={}
 for fighter in ('fighter_black','fighter_red'):
  p=row['fighters'][fighter];variants=next(r for r in new['frames'] if r['clip']==c and r['sample_index_12hz']==48 and r['fighter']==fighter)['variants']
  candidates={'baseline':p['baseline_keypoints'],'opponent_all_removed':p['opponent_masked_keypoints'],**{k:v['keypoints'] for k,v in variants.items()}};panels=[]
  for cond,k in candidates.items():
   pic=im.copy();kp=np.array(k);valid=(kp[:,0]>=0)&(kp[:,0]<960)&(kp[:,1]>=0)&(kp[:,1]<540)&(kp[:,2]>=2);col=COL[fighter]
   for a,b in E:
    if valid[a] and valid[b]:cv2.line(pic,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),col,1,cv2.LINE_AA)
   for j in np.where(valid)[0]:cv2.circle(pic,tuple(kp[j,:2].astype(int)),2,col,-1,cv2.LINE_AA)
   cv2.rectangle(pic,(0,0),(960,40),(16,23,28),-1);cv2.putText(pic,f'{c} t=4s {fighter} {cond}: candidate only',(7,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
   dest=O/f'{c}-048-{fighter}-{cond}.png';cv2.imwrite(str(dest),pic);inputs[c].append(dest);panels.append(pic)
  dest=O/f'{c}-048-{fighter}-comparison.png';cv2.imwrite(str(dest),np.hstack(panels))
  # Actual inference input for all3 masked variants; baseline receives originalpixels.
  inputs[c].append(R/f'outputs/expanded/segmentation/review/{c}-048-{fighter}-masked-input.png')
  inputs[c]+=[R/variants[k]['input'] for k in ['target_mask_only','opponent_exclusive_removed']]
  maps[c][fighter]={'variant_coordinate_sources':['baseline_keypoints','opponent_masked_keypoints','target_mask_only','opponent_exclusive_removed'],'uniform_raw_score_display_cutoff':2}
prompt='''Review ONLY supplied original images, candidate plots, and actual diagnostic inference-input images. No tools, web, files, prior reviews, fighter names, known-fight memories or invisible anatomy inference. This is one selected4secondframe perclip, not temporal tracking. Cyan #22d3ee is Black/gray shorts/light-haired candidate; Coral #fb7185 is Red shorts candidate. Color does NOT establish correctownership. The original frame is authoritative for visibleanatomy. Each candidate plot is drawn on ORIGINALpixels even if model's actual input erased bodyregions. Uniform rawscore threshold2 for allplots; rawscore not calibratedprobability.
Four conditions: baseline uses originalpixels; opponent_all_removed erases the selected whole opponentmask; target_mask_only keeps only selected targetmask pixels; opponent_exclusive_removed erases opponentmask outside targetmask and retains maskintersection. Masks are candidates, not ownershiptruth; sharedmaskpixels may belong to eitherfighter. Neither variantname nor bigger/finerfigure means improvement. Mask boundaries can erase own visiblelimbs or retain oppositefighter. Compare visuallysupported anatomy, never infer correctness just from changed pointpositions or score. Occluded/cropped points remain unknown. A plausible facecluster on the wrong person is a failure. Support may differ by bodypart.
For EACHfighter Black/gray,Red and EACHcondition baseline,opponent_all_removed,target_mask_only,opponent_exclusive_removed: give limbgroup hands/feet/elbows/knees/head/torso status supported|mixed|unclear with brief specificevidence. supported means visiblyaligned with that fighter's own anatomy; mixed means clearlywrong/otherfighter placement or a mixture of correct and unsupportedpoints; unclear means visibility cannot resolve. Then compare each maskedvariant with baseline, identify precise visibleimprovements/harms/erasedownpixels and distinguish allopponentremoval tradeoffs. Do NOT force a winningvariant. No numericaccuracy or populationclaim.
Return JSON {clip,time_seconds:4,assessments:[{fighter_clothing,condition,groups:{hands:{status,evidence},feet:{status,evidence},elbows:{status,evidence},knees:{status,evidence},head:{status,evidence},torso:{status,evidence}}}],comparisons:[{fighter_clothing,condition,effect_vs_baseline:improves|harms|neutral|unclear,visible_improvements:[],visible_harms:[],own_pixel_erasure_evidence,uncertainties:[]}],part_specific_selection:[{fighter_clothing,limbgroup,preferred_candidate:baseline|opponent_all_removed|target_mask_only|opponent_exclusive_removed|null,visible_support,uncertainty}],overall,limitations:[]}. Exactly8assessments and6comparisons. JSONonly.'''
def run(c):
 a=base.call(f'{c}_safer_independent_sol',prompt+'\nReviewclip '+c,inputs[c]);assert len(a['assessments'])==8 and len(a['comparisons'])==6
 for r in a['assessments']:assert set(r['groups'])==set(['hands','feet','elbows','knees','head','torso']) and all(g['status'] in ['supported','mixed','unclear'] for g in r['groups'].values())
 (O/f'{c}-review.json').write_text(json.dumps(a,indent=2));print('REVIEWED',c,flush=True);return a
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:reviews=list(ex.map(run,'CD'))
 (O/'combined-review.json').write_text(json.dumps({'model':'gpt-6.1-sol','reasoning':'high','tool_free':True,'selected_frames':['C4','D4'],'reviews':reviews,'render_only':True,'render_metadata':maps,'sources':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'outputs/expanded/segmentation/sam2-rtmw-c-d-0-4.json',R/'outputs/expanded/segmentation/safer_variants/results.json']}},indent=2));print('COMPLETE',flush=True)
