import json,cv2,numpy as np,concurrent.futures,sys
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded';D=O/'body_data';D.mkdir(exist_ok=True);I=O/'identity';(I/'anchor_audit').mkdir(exist_ok=True);(I/'audit').mkdir(exist_ok=True)
p=json.load(open(O/'poses/E-H.json'))['clips'];M={x['id']:x for x in json.load(open(O/'annotation_manifest.json'))};names={'E':{'fighter_cyan':'Green shorts','fighter_coral':'Yellow shorts'},'F':{'fighter_cyan':'Green shorts','fighter_coral':'Yellow shorts'},'G':{'fighter_cyan':'Black/gray shorts (red wrist tape)','fighter_coral':'Green/black shorts (blue wrist tape)'},'H':{'fighter_cyan':'Gray/black/gold shorts (red wrist tape)','fighter_coral':'Red/black shorts (blue wrist tape)'}}
for c,d in p.items():
 rows=[{'frame':r['sample_index'],'time':r['capture_time_seconds'],'people':[{'box':z['box_xyxy_pixels'],'score':z['detector_score'],'keypoints':z['keypoints_xy_score']} for z in r['people']]} for r in d['frames']];out={'model':'RTMW133','fps':12,'width':d['width'],'height':d['height'],'source':{'path':str(R/M[c]['clip']),'source_manifest':str((O/'annotation_manifest.json').relative_to(R)),'sha256':d['source_sha256']},'frames':rows};(D/f'{c}_rtmw_133.json').write_text(json.dumps(out,separators=(',',':')))
sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as b;b.O=I

def anchor(c):
 d=json.load(open(D/f'{c}_rtmw_133.json'));cap=cv2.VideoCapture(d['source']['path']);tiles=[]
 for fr in range(0,240,24):
  row=d['frames'][fr];cap.set(cv2.CAP_PROP_POS_MSEC,row['time']*1000);ok,im=cap.read();assert ok
  for n,q in enumerate(row['people']):
   x1,y1,x2,y2=map(int,q['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(238,211,34),2);cv2.putText(im,'D'+str(n),(x1,max(28,y1+25)),cv2.FONT_HERSHEY_SIMPLEX,.8,(238,211,34),2)
  im=cv2.resize(im,(640,360));cv2.putText(im,f'{c} sampleindex{fr} t={row["time"]:.3f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);tiles.append(im)
 cap.release();sheet=I/'anchor_audit'/f'{c}-numbered.jpg';cv2.imwrite(str(sheet),np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,10,2)]))
 prompt='UseONLYsuppliedorderedfullframeimage sheet. No tools,files,web,fightmemory. EachframecontainsnumbereddetectorboxesD0,D1,etc. ClassifyclothingidentityforEACHbox EACH10sampleindices 0,24,48,72,96,120,144,168,192,216. Choices:'+json.dumps(names[c])+'. Ifboxmergesbothfighterbodies,couldbereferee/background,oridentity/ownershipunclear, identitynull. DoNOTforce2identities because2boxesexist. Stableclothingrole notscreenposition; distinguish wristtapeonlyvisibleconnectedbody. Return JSON {frames:[{frame:0,detections:[{detection_index:0,identity:fighter_cyan|fighter_coral|null,certainty:high|medium|low,evidence}]}],limitations:[]}. IdentityassignmentdoesNOTverifyoccludedkeypoints.'
 a=b.call(f'{c}_anchor',prompt,[sheet]);(I/'anchor_audit'/f'{c}.json').write_text(json.dumps(a,indent=2));print('ANCHOR',c,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(anchor,'EFGH'))
s=(R/'work/expanded/identity.py').read_text();start=s.index("PALETTE=");end=s.index('EDGES=');s=s[:start]+"PALETTE={'fighter_cyan':'#22d3ee','fighter_coral':'#fb7185'}\nANCHORS={}\nNAMES="+repr(names)+"\n"+s[end:];s=s.replace("for sid in 'ABCD':","for sid in 'EFGH':").replace(" PALETTE['fighter_black']='#fb7185' if sid=='B' else '#22d3ee'\n",'').replace('outputs/experiment/body_data/','outputs/expanded/body_data/').replace("No identity claims for C/D: separate fighter anchors unavailable in initial merged boxes.","Clothingcoloranchors fromseparateSolreview; overlapping/merged/unassigneddetections remain gray.")
s=s.replace("(OUT/'summary.json')","(OUT/'new-summary.json')");(R/'work/expanded/identity_new_generated.py').write_text(s);exec(compile(s,'identity_new_generated.py','exec'))
s=(R/'work/expanded/render_colors.py').read_text().replace("for c in 'ABCD':","for c in 'EFGH':").replace("R/'outputs/experiment/body_data'","R/'outputs/expanded/body_data'").replace("R/'outputs/experiment/clips'","R/'outputs/expanded/clips'").replace('960x600','1280x780').replace('(960,600)','(1280,780)').replace('x]<960','x]<1280').replace('0]<960','0]<1280').replace('1]<540','1]<720').replace('(15,549)','(15,729)').replace('(15,575)','(15,755)').replace("color_videos/validation.json","color_videos/new-validation.json")
(R/'work/expanded/render_new_generated.py').write_text(s);exec(compile(s,'render_new_generated.py','exec'));print('NEW COLOR FIGURES DONE',flush=True)
