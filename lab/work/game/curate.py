from pathlib import Path
import shutil
p=Path('work/game/build_game.py');s=p.read_text();shutil.copy2(p,p.with_name('build_game_initial.py'))
s=s.replace('rounds=[];excluded=[]', '''# A single window per distinct committed attack/exchange, from reviewed native-frame labels.
SELECTION={
'A':[(.25,2.25,.875,1.375,'Punch into front kick: follow the knee lift, extension and retraction.','outputs/experiment/audit/label_A_00.json'),(9.25,2.5,10.25,10.75,'A step-in straight punch, then a return to striking distance.','outputs/experiment/audit/label_A_08.json')],
'E':[(1.1,3.25,2,2.2667,'A low kick meets the forward leg. Watch the short extension and quick reset.','outputs/expanded/annotations/E_detail.json'),(6.9,3.1,7.9,8.4667,'A rising round kick reaches the raised guard at head height.','outputs/expanded/annotations/E_detail.json')],
'B':[(3.6,2.65,4.638,4.905,'A straight punch leads into a kick toward the upper body.','outputs/expanded/key_moments/subtypes/references/B_4.json')],
'P':[(.5,2.5,1.25,1.625,'A body kick, fast retraction, then a withdrawal to reset the distance.','outputs/expanded/mcgregor/dense_check/labels/D01.json'),(2.65,2.35,3.375,3.75,'Another body kick: look at the turn, extension and recovery.','outputs/expanded/mcgregor/dense_check/labels/D01.json'),(5.6,2.4,6.375,6.625,'A head-height punch with torso rotation, followed by retraction.','outputs/expanded/mcgregor/dense_check/labels/D01.json')],
'Q':[(.75,2.75,1.625,2,'A long lunging punch: entry, stance recovery, then withdrawal.','outputs/expanded/mcgregor/dense_check/labels/D02.json')],
'R':[(0,2.1,.25,.75,'A head-height punch extends and retracts before the next reset.','outputs/expanded/mcgregor/expansion/labels/Y07.json'),(2.25,2.75,3,3.75,'A leg lift develops into a kick, then lowers back into stance.','outputs/expanded/mcgregor/expansion/labels/Y07.json'),(5.55,2.45,6.5,7,'A head-height punch with a torso turn and visible retraction.','outputs/expanded/mcgregor/expansion/labels/Y07.json')],
'S':[(4.4,3.2,5.5,6,'A lowered posture becomes a forward punching entry toward the upper body.','outputs/expanded/mcgregor/pilot/labels/X08.json')]
}
rounds=[];excluded=[];selection_audit=[]''')
s=s.replace("starts=list(range(0,17,2)) if duration>10 else [0,2,4];length=4 if duration>10 else 3.5", "candidates=SELECTION.get(cid,[])")
s=s.replace('for start in starts:', 'for start,length,event_start,event_end,insight,evidence in candidates:')
s=s.replace("number=len(rounds)+1;name=", "assert start<=event_start<event_end<=start+length\n  selection_audit.append({'source':cid,'start':start,'duration':length,'strike_start':event_start,'strike_end':event_end,'evidence':evidence,'selection':insight,'both_fighters_coverage':round(both,3)})\n  number=len(rounds)+1;name=")
s=s.replace("'source_id':cid}", "'source_id':cid,'strike_start':round(event_start-start,4),'strike_end':round(event_end-start,4)}")
s += '''\n# Bundle only sources actually used in the curated card.
(O/'footage').mkdir(exist_ok=True)
for index,cid in enumerate(dict.fromkeys(r['source_id'] for r in rounds),1):
 spec=next(s for s in SPECS if s[0]==cid);raw=(O/spec[4]).resolve();dest=O/'footage'/f'v{index:02d}.mp4';shutil.copy2(raw,dest)
 for r in rounds:
  if r['source_id']==cid:r['video']=f'footage/v{index:02d}.mp4'
(O/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
(O/'strike-selection.json').write_text(json.dumps({'criterion':'Distinct committed punch or kick visible in reviewed frames, with preparation and recovery. Quiet probes, ambiguous attempts and reset-only windows excluded. Important means a substantive attack, not proven impact or damage.','rounds':selection_audit},indent=2))
'''
p.write_text(s)
