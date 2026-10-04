from reviewed_predict import SCHEMA
from run import *
from colors import colorize
MAP={'A':'01-standing','B':'02-takedown','C':'03-ground','D':'04-finish'}
LANDMARKS={0:'nose',1:'left_eye',2:'right_eye',3:'left_ear',4:'right_ear',5:'left_shoulder',6:'right_shoulder',7:'left_elbow',8:'right_elbow',9:'left_wrist',10:'right_wrist',11:'left_hip',12:'right_hip',13:'left_knee',14:'right_knee',15:'left_ankle',16:'right_ankle',17:'left_big_toe',18:'left_small_toe',19:'left_heel',20:'right_big_toe',21:'right_small_toe',22:'right_heel',23:'face_outline_start',31:'chin',39:'face_outline_end',53:'face_nose_bridge',59:'left_eye_contour',65:'right_eye_contour',71:'mouth_outer_corner',77:'mouth_opposite_corner',91:'left_hand_wrist',95:'left_thumb_tip',99:'left_index_tip',103:'left_middle_tip',107:'left_ring_tip',111:'left_pinky_tip',112:'right_hand_wrist',116:'right_thumb_tip',120:'right_index_tip',124:'right_middle_tip',128:'right_ring_tip',132:'right_pinky_tip'}
def task(pair):
 c,t=pair;data=[]
 for st in range(0,t,4):
  p=OUT/'adjudication'/f'{c}_{st:02d}.json';p=p if p.exists() else OUT/'verification'/f'{c}_{st:02d}.json';x=colorize(json.loads(p.read_text()));data.append({k:v for k,v in x.items() if k not in ['adjudication_notes','first_action','first_action_time','first_action_evidence']})
 pose=json.loads((ROOT/'outputs/pose'/f'{MAP[c]}__rtmw.json').read_text());frames=[]
 for j in range(8):
  tm=t-4+j*.5;fr=pose['frames'][int(tm*12)];assert fr['time']<t
  people=[]
  for person in fr['people']:
   points={str(k):[round(person['keypoints'][k][0]/960,3),round(person['keypoints'][k][1]/540,3),round(person['keypoints'][k][2],2)] for k in LANDMARKS}
   people.append(dict(clothing_color='unresolved',box=[round(v/(960 if n%2==0 else 540),3) for n,v in enumerate(person['box'])],landmarks=points))
  frames.append(dict(time=fr['time'],detections=people))
 features=dict(landmark_map=LANDMARKS,coordinate_units='x/960, y/540, raw RTMW peak score (NOT a probability)',quality_limits='No persistent identities or proven ownership; detection order may change; crowded ground poses may combine fighters; out-of-bounds coordinates are unobservable; projected joint proximity cannot establish contact; face/hand/foot estimates are unverified.',frames=frames)
 prompt='You are forecasting an anonymous MMA sequence. You must use ONLY the supplied annotation record available at the cutoff. No tools, images, web, fighter identities, fight memories, external data, or future events. These are model-generated annotations and may contain errors. Predict the phase AT THE END of the next four seconds and whether each event occurs ANYWHERE within those next four seconds. Give calibrated probabilities, accounting for continuation/persistence and uncertainty; do not turn a plausible story into certainty. Return valid JSON without markdown.\n'+f'Cutoff={t}s; forecast interval=[{t},{t+4})s; condition=reviewed_pose_history.\nAnnotation prefix:\n'+json.dumps(data)+'\nAdditional recent whole-body estimated annotations (optional evidence, may be wrong):\n'+json.dumps(features)+'\n'+SCHEMA
 tag=f'predict_{c}_{t:02d}_reviewed_pose_history_r1';obj=call(tag,prompt);obj.update(clip=c,cutoff=t,horizon=4,condition='reviewed_pose_history',repeat=1,input_sha256=hashlib.sha256(json.dumps(dict(labels=data,pose=features),sort_keys=True).encode()).hexdigest());(OUT/'predictions'/f'{tag}.json').write_text(json.dumps(obj,indent=2));(OUT/'audit'/f'{tag}.pose_input.json').write_text(json.dumps(features,indent=2));print('POSE PREDICT',c,t,flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(task,[(c,t) for c in 'ABCD' for t in [4,8,12,16]]))
