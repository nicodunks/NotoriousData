from run import *
EV=['strike_attempt','takedown_attempt','ground_position_change']
def task(pair):
 cid,st=pair;orig=[x for x in json.loads((OUT/'labels'/f'{cid}.json').read_text()) if x['start']==st][0];review=json.loads((OUT/'verification'/f'{cid}_{st:02d}.json').read_text())
 prompt=RULES+'\nYou are a third visual adjudicator resolving two MODEL-generated annotations. Neither is ground truth. Inspect ordered native-frame-rate frames; do not vote or trust confident language. Return a corrected complete annotation in the given schema, retaining unclear wherever not visually resolved. Distinguish hand-fighting from committed strikes, and a maintained position from meaningful ground advancement. Do NOT speculate about later actions.\nInterval='+str((st,st+4))+'\nFirst causal pass:\n'+json.dumps(orig)+'\nIndependent dense review:\n'+json.dumps(review)+'\n'+LABEL_TEMPLATE+' Also include first_action, first_action_time, first_action_evidence with categories punch/kick_knee_elbow/takedown_entry/ground_strike/ground_control_change/none/unclear, and adjudication_notes describing each material correction or unresolved disagreement. First action means first observable committed action; ongoing discrete execution counts, maintained positions/grips alone do not; simultaneous different action categories with unresolved ordering are unclear. End phase is last observed frame before boundary. For each strike event add visible_contact_time (number or null) and contact_surface (head/body/leg/guard/unclear/not_applicable). A timestamp is permitted ONLY for the first supplied frame that clearly supports actual contact with fighter or guard; trajectory proximity/overlap alone is insufficient. State the frame-specific support in evidence. For missed or visually unresolved contact, visible_contact_time must be null. This does not measure force or prove an unseen impact.'
 obj=call(f'adjudicate_{cid}_{st:02d}',prompt,[BASE/'native'/cid/f'{j:02d}.jpg' for j in range(st,st+4)]);obj.update(start=st,end=st+4)
 (OUT/'adjudication'/f'{cid}_{st:02d}.json').write_text(json.dumps(obj,indent=2));print('ADJUDICATE',cid,st,flush=True)
if __name__=='__main__':
 (OUT/'adjudication').mkdir(exist_ok=True);chosen=[]
 for c in 'ABCD':
  for orig in json.loads((OUT/'labels'/f'{c}.json').read_text()):
   st=orig['start'];v=json.loads((OUT/'verification'/f'{c}_{st:02d}.json').read_text())
   # Predefined triggers: any target disagreement/unknown, or any claimed definite strike contact requiring review.
   reasons=[]
   if orig['end_phase']!=v['end_phase']:reasons.append('phase disagreement')
   for e in EV:
    if orig['event_flags'][e]!=v['event_flags'][e]:reasons.append(e+' disagreement')
   if v.get('first_action')=='unclear':reasons.append('first action unclear')
   if any(e['target'] in ['head','body','leg'] and e['contact']!='not_applicable' for e in orig['events']+v['events']):reasons.append('strike/contact candidate, including unclear')
   if any(w in e['action'].lower() for e in orig['events']+v['events'] for w in ['choke','submission']):reasons.append('submission candidate')
   if reasons:chosen.append((c,st));
   (OUT/'adjudication'/f'{c}_{st:02d}.trigger.json').write_text(json.dumps(dict(clip=c,start=st,reasons=reasons,selected=bool(reasons)),indent=2))
 print('ADJUDICATION SELECTED',chosen,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(task,chosen))
