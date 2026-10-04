import json,hashlib,shutil
from pathlib import Path
R=Path.cwd();O=R/'outputs/game';O.mkdir(exist_ok=True);(O/'motion').mkdir(exist_ok=True)
SPECS=[('A',R/'outputs/experiment/body_data/A_rtmw_133.json',R/'outputs/expanded/identity/A-identities.json',{'fighter_green':'Alex Pereira','fighter_yellow':'Israel Adesanya'},'../experiment/clips/A.mp4',None,'Long-limbed standing exchanges. Watch how each fighter manages the space.'),('E',R/'outputs/expanded/body_data/E_rtmmw_133.json',R/'outputs/expanded/identity/E-identities.json',{'fighter_cyan':'Alex Pereira','fighter_coral':'Israel Adesanya'},'../expanded/clips/E.mp4',None,'Hand positioning and distance changes are visible; contact often remains unclear.'),('B',R/'outputs/experiment/body_data/B_rtmw_133.json',R/'outputs/expanded/identity/B-identities.json',{'fighter_green':'Conor McGregor','fighter_black':'Khabib Nurmagomedov'},'../experiment/clips/B.mp4',2018,'Before the bodies overlap, the standing movement is easier to recognize.'),('F',R/'outputs/expanded/body_data/F_rtmw_133.json',R/'outputs/expanded/identity/F-identities.json',{'fighter_cyan':'Alex Pereira','fighter_coral':'Israel Adesanya'},'../expanded/clips/F.mp4',None,'The fence exchange makes limb ownership harder. Missing tracks stay missing.')]
SPECS[1]=tuple(str(x).replace('rtmmw','rtmw') if isinstance(x,str) else Path(str(x).replace('rtmmw','rtmw')) if isinstance(x,Path) else x for x in SPECS[1])
for c,year,opp,insight in [('P',2012,'Ivan Buchinger','Kicks, retraction and distance resets give this exchange its rhythm.'),('Q',2013,'Max Holloway','Shuffle → long punching entry → stance recovery → withdrawal.'),('R',2018,'Khabib Nurmagomedov','Short repositioning and hand probing; foot visibility limits the reading.'),('S',2021,'Dustin Poirier','Punch-led pursuit, with repeated forward steps toward the fence.')]:
 base=R/'outputs/expanded/mcgregor/tracked';SPECS.append((c,base/'body_data'/f'{c}_rtmw_133.json',base/'identity'/f'{c}-identities.json',{'fighter_cyan':'Conor McGregor','fighter_coral':opp},f'../expanded/mcgregor/four_bout/clips/{year}-native-standing.mp4',year,insight))
# A single window per distinct committed attack/exchange, from reviewed native-frame labels.
SELECTION={
'A':[(.25,2.25,.875,1.375,'Punch into front kick: follow the knee lift, extension and retraction.','outputs/experiment/audit/label_A_00.json'),(9.25,2.5,10.25,10.75,'A step-in straight punch, then a return to striking distance.','outputs/experiment/audit/label_A_08.json')],
'E':[(1.1,3.25,2,2.2667,'A low kick meets the forward leg. Watch the short extension and quick reset.','outputs/expanded/annotations/E_detail.json'),(6.9,3.1,7.9,8.4667,'A rising round kick reaches the raised guard at head height.','outputs/expanded/annotations/E_detail.json')],
'B':[(3.6,2.65,4.638,4.905,'A straight punch leads into a kick toward the upper body.','outputs/expanded/key_moments/subtypes/references/B_4.json')],
'P':[(.5,2.5,1.25,1.625,'A body kick, fast retraction, then a withdrawal to reset the distance.','outputs/expanded/mcgregor/dense_check/labels/D01.json'),(2.65,2.35,3.375,3.75,'Another body kick: look at the turn, extension and recovery.','outputs/expanded/mcgregor/dense_check/labels/D01.json'),(5.6,2.4,6.375,6.625,'A head-height punch with torso rotation, followed by retraction.','outputs/expanded/mcgregor/dense_check/labels/D01.json')],
'Q':[(.75,2.75,1.625,2,'A long lunging punch: entry, stance recovery, then withdrawal.','outputs/expanded/mcgregor/dense_check/labels/D02.json')],
'R':[(0,2.1,.25,.75,'A head-height punch extends and retracts before the next reset.','outputs/expanded/mcgregor/expansion/labels/Y07.json'),(2.25,2.75,3,3.75,'A leg lift develops into a kick, then lowers back into stance.','outputs/expanded/mcgregor/expansion/labels/Y07.json'),(5.55,2.45,6.5,7,'A head-height punch with a torso turn and visible retraction.','outputs/expanded/mcgregor/expansion/labels/Y07.json')],
'S':[(4.4,3.2,5.5,6,'A lowered posture becomes a forward punching entry toward the upper body.','outputs/expanded/mcgregor/pilot/labels/X08.json')]
}
rounds=[];excluded=[];selection_audit=[]
for cid,posepath,idpath,names,video,year,insight in SPECS:
 p=json.loads(posepath.read_text());ids=json.loads(idpath.read_text());keys=list(names);duration=p['frames'][-1]['time']+1/p['fps'];candidates=SELECTION.get(cid,[])
 for start,length,event_start,event_end,insight,evidence in candidates:
  subset=[(j,f) for j,f in enumerate(p['frames']) if start<=f['time']<min(duration,start+length)];both=sum(len({z['identity'] for z in ids['frames'][j]['people'] if z['identity'] in keys})==2 for j,f in subset)/len(subset)
  attack_rows=[(j,f) for j,f in subset if event_start<=f['time']<=event_end];attack_both=sum(len({z['identity'] for z in ids['frames'][j]['people'] if z['identity'] in keys})==2 for j,f in attack_rows)/len(attack_rows)
  if both<.74 or attack_both<.74:excluded.append({'source':cid,'start':start,'both_fighters_coverage':both,'strike_frame_coverage':attack_both});continue
  assert start<=event_start<event_end<=start+length
  selection_audit.append({'source':cid,'start':start,'duration':length,'strike_start':event_start,'strike_end':event_end,'evidence':evidence,'selection':insight,'both_fighters_coverage':round(both,3)})
  number=len(rounds)+1;name=f'r{number:02d}';frames=[];allxy=[]
  for j,f in subset:
   people=[]
   for z in ids['frames'][j]['people']:
    if z['identity'] not in keys:continue
    k=f['people'][z['detection_index']]['keypoints'];pts=[[round(x,1),round(y,1)] if sc>=2 and 0<=x<p['width'] and 0<=y<p['height'] else None for x,y,sc in k];allxy.extend(q for q in pts[:23] if q);people.append({'role':keys.index(z['identity']),'points':pts})
   frames.append({'t':round(f['time']-start,4),'people':people})
  assert allxy;xs,ys=zip(*allxy);bbox=[max(0,min(xs)-30),max(0,min(ys)-30),min(p['width'],max(xs)+30),min(p['height'],max(ys)+30)];motion={'width':p['width'],'height':p['height'],'duration':round(min(length,duration-start),4),'fps':p['fps'],'bounds':bbox,'frames':frames};(O/'motion'/f'{name}.json').write_text(json.dumps(motion,separators=(',',':')));rounds.append({'id':name,'motion':f'motion/{name}.json','names':[names[k] for k in keys],'video':video,'start':start,'duration':motion['duration'],'year':year,'insight':insight,'quality':round(both,3),'source_id':cid,'strike_start':round(event_start-start,4),'strike_end':round(event_end-start,4)})
manifest={'rounds':rounds,'roster':sorted({n for r in rounds for n in r['names']}),'peter':'https://pwang724.github.io/tennis-skeleton-quiz/','limits':'Recognition game from a small repeated-bout collection. Color swaps per round. Missing/ambiguous identities excluded; same-model anchors are not perfect identity truth.'};(O/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')));(O/'build-audit.json').write_text(json.dumps({'rounds':len(rounds),'sources':len(SPECS),'excluded_windows':excluded,'gate':'Both clothing roles associated in >=74% of sampled frames; candidate points raw score>=2 and in image bounds','source_hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for s in SPECS for p in [s[1],s[2]]}},indent=2));print('GAME POOL',len(rounds),'rounds',len(manifest['roster']),'fighters',len([r for r in rounds if r['source_id'] in 'PQRS']),'era windows')

# Bundle only sources actually used in the curated card.
(O/'footage').mkdir(exist_ok=True)
for index,cid in enumerate(dict.fromkeys(r['source_id'] for r in rounds),1):
 spec=next(s for s in SPECS if s[0]==cid);raw=(O/spec[4]).resolve();dest=O/'footage'/f'v{index:02d}.mp4';shutil.copy2(raw,dest)
 for r in rounds:
  if r['source_id']==cid:r['video']=f'footage/v{index:02d}.mp4'
(O/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
(O/'strike-selection.json').write_text(json.dumps({'criterion':'Distinct committed punch or kick visible in reviewed frames, with preparation and recovery. Quiet probes, ambiguous attempts and reset-only windows excluded. Important means a substantive attack, not proven impact or damage.','rounds':selection_audit},indent=2))
