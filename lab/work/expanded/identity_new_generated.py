"""Conservative appearance + temporal association. No person recognition or training.
Input is frozen 12Hz RTMW output; abstains on merged/ambiguous boxes.
"""
import json,itertools
from pathlib import Path
import cv2,numpy as np
ROOT=Path.cwd();OUT=ROOT/'outputs/expanded/identity';OUT.mkdir(parents=True,exist_ok=True)
PALETTE={'fighter_cyan':'#22d3ee','fighter_coral':'#fb7185'}
ANCHORS={}
NAMES={'E': {'fighter_cyan': 'Green shorts', 'fighter_coral': 'Yellow shorts'}, 'F': {'fighter_cyan': 'Green shorts', 'fighter_coral': 'Yellow shorts'}, 'G': {'fighter_cyan': 'Black/gray shorts (red wrist tape)', 'fighter_coral': 'Green/black shorts (blue wrist tape)'}, 'H': {'fighter_cyan': 'Gray/black/gold shorts (red wrist tape)', 'fighter_coral': 'Red/black shorts (blue wrist tape)'}}
EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for root in (91,112):
 for j in range(5):
  chain=[root]+list(range(root+1+j*4,root+5+j*4));EDGES+=list(zip(chain,chain[1:]))

def descriptor(im,b):
 x1,y1,x2,y2=np.array(b).astype(int);h,w=im.shape[:2];roi=im[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
 if roi.size==0:return np.zeros(24)
 hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],None,[12,2],[0,180,0,256]).flatten();return hist/(hist.sum()+1e-8)
def iou(a,b):
 a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return inter/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter+1e-8)
def bgr(key):
 h=PALETTE[key].lstrip('#');return tuple(int(h[i:i+2],16) for i in (4,2,0))
summary={}
for sid in 'EFGH':
 data=json.load(open(ROOT/f'outputs/expanded/body_data/{sid}_rtmw_133.json'));cap=cv2.VideoCapture(data['source']['path']);states={};rows=[];panels=[];counts={}
 ap=OUT/'anchor_audit'/f'{sid}.json';anchor_rows={r['frame']:r for r in json.load(open(ap))['frames']} if ap.exists() else {}
 for row in data['frames']:
  cap.set(cv2.CAP_PROP_POS_MSEC,row['time']*1000);ok,im=cap.read()
  if not ok:break
  desc=[descriptor(im,p['box']) for p in row['people']];assigned={};reason='no reliable separated anchor'
  if row['frame'] in anchor_rows:
   for a in anchor_rows[row['frame']]['detections']:
    j=a['detection_index'];key=a['identity']
    if key in NAMES[sid] and a['certainty']=='high' and j<len(desc) and len(desc)>1 and max(iou(row['people'][j]['box'],p['box']) for q,p in enumerate(row['people']) if q!=j)<.5:assigned[j]=key
   reason='fresh tool-free Sol visible-clothing anchor; overlap gate passed'
  elif row['frame']==0 and sid in ANCHORS and len(desc)==2:
   assigned={j:key for j,key in enumerate(ANCHORS[sid])};reason='visually reviewed first-frame clothing anchor'
  elif sid=='A' and row['frame'] in (120,180,228) and len(desc)==2:
   order=sorted(range(2),key=lambda j:row['people'][j]['box'][0]);assigned={order[0]:'fighter_green',order[1]:'fighter_yellow'};reason='visually reviewed contact-sheet clothing reanchor'
  elif states and len(desc)>=2:
   keys=list(states);choices=[]
   for perm in itertools.permutations(range(len(desc)),len(keys)):
    costs=[]
    for key,j in zip(keys,perm):
     st=states[key];dist=np.linalg.norm((np.array(row['people'][j]['box'][:2])+row['people'][j]['box'][2:])/2-st['center'])/data['width']
     costs.append(.65*(1-iou(st['box'],row['people'][j]['box']))+.8*np.abs(st['appearance']-desc[j]).sum()+dist)
    choices.append((sum(costs),perm,costs))
   choices.sort();best=choices[0];margin=(choices[1][0]-best[0]) if len(choices)>1 else 99
   if margin>.22:
    for key,j,c in zip(keys,best[1],best[2]):
     age=row['frame']-states[key]['frame']
     if c<.85 and age<=12 and max(iou(row['people'][j]['box'],p['box']) for q,p in enumerate(row['people']) if q!=j)<.5:assigned[j]=key
   reason='appearance/motion gate passed' if assigned else 'association ambiguous, overlapping, or stale'
  people=[]
  for j,p in enumerate(row['people']):
   key=assigned.get(j);counts[key or 'unassigned']=counts.get(key or 'unassigned',0)+1
   b=np.array(p['box']);people.append({'detection_index':j,'box':p['box'],'identity':key,'display_color':PALETTE.get(key,'#a3a3a3'),'identity_status':'anchored' if key else 'unassigned','reason':reason if key else ('single box may merge both fighters' if len(desc)==1 else reason),'keypoint_ownership':'candidate detection ownership; occluded joints unverified' if key else 'unknown','association_score_is_probability':False})
   if key:
    prev=states.get(key);states[key]={'box':b,'center':(b[:2]+b[2:])/2,'appearance':desc[j] if prev is None else .9*prev['appearance']+.1*desc[j],'frame':row['frame']}
   col=bgr(key) if key else (163,163,163);cv2.rectangle(im,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),col,2)
   cv2.putText(im,NAMES[sid].get(key,'Unassigned / merged?'),tuple((b[:2]+[0,18]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1,cv2.LINE_AA)
   k=np.array(p['keypoints'])
   # Raw SIMCC score is not calibrated visibility: render only in-bounds candidates.
   valid=(k[:,0]>=0)&(k[:,0]<data['width'])&(k[:,1]>=0)&(k[:,1]<data['height'])&(k[:,2]>=2)
   for a,z in EDGES:
    if valid[a] and valid[z]:cv2.line(im,tuple(k[a,:2].astype(int)),tuple(k[z,:2].astype(int)),col,1,cv2.LINE_AA)
   for q in np.where(valid)[0]:cv2.circle(im,tuple(k[q,:2].astype(int)),1,col,-1)
  rows.append({'frame':row['frame'],'time':row['time'],'people':people})
  if row['frame'] in (0,24,72,120,180,228):
   tile=cv2.resize(im,(640,360));cv2.putText(tile,f'{sid} {row["time"]:.2f}s | candidates, not visible-joint truth',(8,350),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);panels.append(tile)
 cap.release();cv2.imwrite(str(OUT/f'{sid}-identity-contact.jpg'),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
 result={'clip':sid,'source':data['source'],'fps':data['fps'],'identities':{k:{'label':v,'color':PALETTE[k]} for k,v in NAMES[sid].items()},'method':'reviewed A/B clothing anchors; gated appearance + temporal box association; abstain on missing/merged/ambiguous detections','limitations':['Clothingcoloranchors fromseparateSolreview; overlapping/merged/unassigneddetections remain gray.','Persistent colors identify clothing roles within a clip, not recognized personal names.','Raw RTMW scores exceed 1 and are not calibrated visibility probabilities.','All face, hand, foot keypoints are candidates; occlusion and limb ownership require separate verification.'],'counts':counts,'frames':rows}
 (OUT/f'{sid}-identities.json').write_text(json.dumps(result,separators=(',',':')));summary[sid]=counts
(OUT/'new-summary.json').write_text(json.dumps(summary,indent=2));print(summary)
