from pathlib import Path
import json,cv2,numpy as np,sys,time,onnxruntime as ort
from rtmlib import RTMPose
R=Path.cwd();O=R/'outputs/expanded/segmentation/safer_variants';O.mkdir(exist_ok=True)
old=json.load(open(R/'outputs/expanded/segmentation/sam2-rtmw-c-d-0-4.json'));model=RTMPose(old['software']['rtmw_checkpoint'],model_input_size=(288,384),backend='onnxruntime',device='cpu');opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1;model.session=ort.InferenceSession(old['software']['rtmw_checkpoint'],sess_options=opts,providers=['CPUExecutionProvider'])
COLS={'fighter_black':(238,211,34),'fighter_red':(133,113,251)};EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for root in [91,112]:
 for j in range(5):
  chain=[root]+list(range(root+1+j*4,root+5+j*4));EDGES+=list(zip(chain,chain[1:]))
def render(im,k,col):
 a=im.copy();k=np.array(k);v=(k[:,0]>=0)&(k[:,0]<im.shape[1])&(k[:,1]>=0)&(k[:,1]<im.shape[0])&(k[:,2]>1.5)
 for x,y in EDGES:
  if v[x] and v[y]:cv2.line(a,tuple(k[x,:2].astype(int)),tuple(k[y,:2].astype(int)),col,2,cv2.LINE_AA)
 for j in np.where(v)[0]:cv2.circle(a,tuple(k[j,:2].astype(int)),1,col,-1,cv2.LINE_AA)
 return a
rows=[];thumbs=[]
for r in old['frames']:
 cap=cv2.VideoCapture(str(R/r['input']));cap.set(cv2.CAP_PROP_POS_MSEC,r['time_seconds']*1000);ok,im=cap.read();cap.release();assert ok
 masks={k:cv2.imread(str(R/p['mask_path']),0)>0 for k,p in r['fighters'].items()};bg=np.median(im.reshape(-1,3),axis=0).astype('uint8')
 for k,p in r['fighters'].items():
  opp='fighter_red' if k=='fighter_black' else 'fighter_black';box=np.array(p['box'],np.float32);target=im.copy();target[~masks[k]]=bg;safe=im.copy();safe[masks[opp]&~masks[k]]=bg
  result={'clip':r['clip'],'sample_index_12hz':r['sample_index_12hz'],'time_seconds':r['time_seconds'],'fighter':k,'display_color':'#22d3ee' if k=='fighter_black' else '#fb7185','variants':{}}
  kpbase=p['baseline_keypoints'];imgs=[im.copy(),render(im,kpbase,COLS[k])];names=['Original pixels','Baseline pose candidates']
  for name,inp in [('target_mask_only',target),('opponent_exclusive_removed',safe)]:
   t=time.time();xy,sc=model(inp,[box]);kp=np.column_stack((xy[0],sc[0]));path=O/f"{r['clip']}-{r['sample_index_12hz']:03d}-{k}-{name}-input.jpg";cv2.imwrite(str(path),inp)
   result['variants'][name]={'keypoints':kp.round(3).tolist(),'seconds':time.time()-t,'input':str(path.relative_to(R)),'median_displacement_23_from_baseline':float(np.median(np.linalg.norm(kp[:23,:2]-np.array(kpbase)[:23,:2],axis=1))),'display_limits':'Candidatepointpositions only; erased anatomy does notprovehiddenjointlocation; rawSIMCCscores uncalibrated'};imgs.append(render(im,kp,COLS[k]));names.append(name.replace('_',' '))
  for a,n in zip(imgs,names):cv2.rectangle(a,(0,0),(im.shape[1],35),(16,23,28),-1);cv2.putText(a,n,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
  sheet=np.hstack(imgs);path=O/f"{r['clip']}-{r['sample_index_12hz']:03d}-{k}-comparison.jpg";cv2.imwrite(str(path),sheet);result['comparison']=str(path.relative_to(R));rows.append(result);thumbs.append(cv2.resize(sheet,(1440,203)));print(r['clip'],r['time_seconds'],k,flush=True)
cv2.imwrite(str(O/'contact-sheet.jpg'),np.vstack(thumbs));(O/'results.json').write_text(json.dumps({'model':'Pretrained RTMW133 samecheckpointasSAMdiagnostic','no_training':True,'mask_source':'../sam2-rtmw-c-d-0-4.json','variants':'Targetmaskonly retains sharedpixels inside ownmask; opponentexclusive removal erases only opponentmask outside targetmask, preserves uncertain overlap. Neither model output isownershiptruth.','frames':rows},indent=2));print('SAFER VARIANTS DONE',flush=True)
