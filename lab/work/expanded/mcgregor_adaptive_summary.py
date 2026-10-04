import json,math,numpy as np
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/adaptive';cases=json.load(open(O/'registered_supplement.json'))['cases'];measurements=[];gates=[]
for case in cases:
 cid=case['id'];lp=O/'labels'/f'{cid}.json';mp=O/'audit'/f'{cid}.metadata.json'
 if not lp.exists() or not mp.exists():continue
 meta=json.load(open(mp));assert not meta['tools_used'];review=json.load(open(lp));pose=json.load(open(O/'poses'/f'{cid}.json'))
 for r in review['frames']:
  reasons=[];t=r['local_time'];row=next((p for p in pose['frames'] if abs(p['local_time']-t)<1e-5),None)
  if not any(q['start']<=t<q['end'] for q in case['quiet_intervals']):reasons.append('not inside registered quiet interval')
  if not (r['standing'] and r['quiet'] and r['feet_visible']):reasons.append('standing/quiet/feet gate rejected')
  if r['target_identity']!='high':reasons.append('identity not high')
  if r['foreshortening']=='high':reasons.append('high foreshortening')
  j=r['target_detection_index'];person=next((p for p in row['people'] if p['detection_index']==j),None) if row else None
  if person is None:reasons.append('target detection unresolved')
  gates.append({'case':cid,'time':t,'base_passed':not reasons,'reasons':reasons,'accepted_landmarks':r['accepted_landmarks'],'evidence':r['evidence']})
  if reasons:continue
  k=np.array(person['keypoints'])[:,:2];a=r['accepted_landmarks'];m={'case':cid,'source':case['source'],'source_time':case['start']+t,'view':r['view'],'foreshortening':r['foreshortening'],'heel_contact_visible_label':r['heel_contact'],'method':'RTMW image coordinates, landmarks checked against original pixels by fresh tool-free Sol; exploratory candidate geometry, not gold truth','values':{}}
  if all(a.get(q,False) for q in ['LA','RA']):
   span=float(np.linalg.norm(k[15]-k[16]))
   for sh,ank,si,ai in [('LS','LA',5,15),('RS','RA',6,16)]:
    if a.get(sh,False):
     scale=float(np.linalg.norm(k[si]-k[ai]))
     if scale>30:m['values']['projected_ankle_span_per_'+sh+'_to_'+ank]=round(span/scale,3)
  if all(a.get(q,False) for q in ['N','LA','RA']):
   ankle_mid=(k[15]+k[16])/2;height=np.linalg.norm(ankle_mid-k[0])
   if height>30:
    m['values']['projected_ankle_span_per_nose_to_ankle_midpoint']=round(float(np.linalg.norm(k[15]-k[16])/height),3)
    for q,idx in [('LW',9),('RW',10)]:
     if a.get(q,False):m['values'][q+'_projected_head_distance_per_body_height_proxy']=round(float(np.linalg.norm(k[idx]-k[0])/height),3)
  if all(a.get(q,False) for q in ['LH','RH','LA','RA']):
   leg=(np.linalg.norm(k[11]-k[15])+np.linalg.norm(k[12]-k[16]))/2
   if leg>20:m['values']['projected_ankle_span_per_projected_leg']=round(float(np.linalg.norm(k[15]-k[16])/leg),3)
  if all(a.get(q,False) for q in ['LS','RS','LH','RH']):
   sh=(k[5]+k[6])/2;hip=(k[11]+k[12])/2;torso=np.linalg.norm(hip-sh)
   if torso>20:
    m['values']['torso_angle_from_image_vertical_degrees']=round(math.degrees(math.atan2(hip[0]-sh[0],hip[1]-sh[1])),2)
    for q,idx in [('LW',9),('RW',10)]:
     if a.get(q,False):
      m['values'][q+'_vertical_below_shoulders_per_torso']=round(float((k[idx,1]-sh[1])/torso),3)
      if a.get('N',False):m['values'][q+'_projected_head_distance_per_torso']=round(float(np.linalg.norm(k[idx]-k[0])/torso),3)
  if m['values']:measurements.append(m)
(O/'validated_measurements.json').write_text(json.dumps({'measurements':measurements,'gates':gates,'interpretation':'Conditional quiet, clear-view supplement. Per-frame descriptive image-plane metrics only; no aggregate career direction or 3D/force/bodyweight inference. Lead/rear wrist roles remain unspecified. Raw model scores never used as visibility truth.'},indent=2));print(json.dumps(measurements,indent=2))
