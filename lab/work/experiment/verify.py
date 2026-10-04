from run import *
def verify(task):
 cid,st=task
 imgs=[BASE/'dense'/cid/f'{j:02d}.jpg' for j in range(st,st+4)]
 prompt=RULES+'\nYou are an independent visual reviewer. No original annotations are supplied. Annotate ONLY this 4-second interval from denser 16-Hz frames. Identify each genuine strike attempt, but distinguish hand-fighting/frames/feints from punches. Include failed attempts and noncontact. Do not manufacture target/contact evidence. Be conservative about submission names, committed takedowns, clean impact and ground-position changes.\nInterval='+str((st,st+4))+'\n'+LABEL_TEMPLATE+' Also add first_action with one of punch/kick_knee_elbow/takedown_entry/ground_strike/ground_control_change/none/unclear: FIRST committed observable action in this interval, excluding probes/hand-fighting/feints. Include first_action_time (number or null) and first_action_evidence. An ongoing discrete attack counts if its execution continues in this interval; maintained grips/positions alone do not count. If different action classes occur simultaneously within one sample step and order cannot be established, first_action is unclear. End phase is the last observable sampled state before the interval boundary, not an unseen exact endpoint. Ground punches use ground_strike; control change must be meaningful advance/reversal; if uncertain choose unclear.'
 obj=call(f'verify_{cid}_{st:02d}',prompt,imgs);obj.update(start=st,end=st+4)
 (OUT/'verification'/f'{cid}_{st:02d}.json').write_text(json.dumps(obj,indent=2))
 with LOCK:print('VERIFY',cid,st+4,flush=True)
if __name__=='__main__':
 tasks=[(c,t) for c in 'ABCD' for t in range(0,20,4)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(verify,tasks))
 print('INDEPENDENT REVIEW COMPLETE',flush=True)
