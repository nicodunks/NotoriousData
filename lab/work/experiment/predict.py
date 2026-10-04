from run import *
SCHEMA='''Return ONLY JSON: {"phase_probs":{"standing":0.25,"clinch":0.25,"ground":0.25,"unclear":0.25},"event_probs":{"strike_attempt":0.5,"takedown_attempt":0.5,"ground_position_change":0.5},"next_action_probs":{"punch":0.14,"kick_knee_elbow":0.14,"takedown_entry":0.14,"ground_strike":0.14,"ground_control_change":0.14,"none":0.15,"unclear":0.15},"forecast":"concise concrete expected continuation; do not assert certainty","evidence":["only facts present in supplied annotations"],"limits":["important missing information"]}. Phase probabilities and next_action_probs must each sum to 1. next_action is the FIRST committed observable action in the future interval (hand-fighting and feints do not count; ground punches belong to ground_strike; meaningful ground control advance/reversal belongs to ground_control_change; none if no committed action). Event probabilities independent in [0,1]. Ground position change means meaningful change of ground position/control, not every adjustment; strikes include ground strikes; takedown requires committed entry/trip/throw rather than feint alone.'''
def case(task):
 cid,cutoff,cond,rep=task;hist=json.loads((OUT/'labels'/f'{cid}.json').read_text());prefix=[x for x in hist if x['end']<=cutoff]
 if cond=='coarse_history':data=[{k:x[k] for k in ['start','end','start_phase','end_phase','visibility']} for x in prefix]
 elif cond=='rich_recent':data=prefix[-1:]
 else:data=prefix
 prompt='''You are forecasting an anonymous MMA sequence. You must use ONLY the supplied annotation record available at the cutoff. No tools, images, web, fighter identities, fight memories, external data, or future events. These are model-generated annotations and may contain errors. Predict the phase AT THE END of the next four seconds and whether each event occurs ANYWHERE within those next four seconds. Give calibrated probabilities, accounting for continuation/persistence and uncertainty; do not turn a plausible story into certainty. Return valid JSON without markdown.\n'''+f'Cutoff={cutoff}s; forecast interval=[{cutoff},{cutoff+4})s; condition={cond}.\nAnnotation prefix:\n'+json.dumps(data)+'\n'+SCHEMA
 tag=f'predict_{cid}_{cutoff:02d}_{cond}_r{rep}';obj=call(tag,prompt)
 assert abs(sum(obj['phase_probs'].values())-1)<.02
 assert abs(sum(obj['next_action_probs'].values())-1)<.02
 assert all(0<=v<=1 for d in ['phase_probs','event_probs','next_action_probs'] for v in obj[d].values())
 obj.update(clip=cid,cutoff=cutoff,horizon=4,condition=cond,repeat=rep,input_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest())
 (OUT/'predictions'/f'{tag}.json').write_text(json.dumps(obj,indent=2))
 with LOCK:print('PREDICT',tag,flush=True)
if __name__=='__main__':
 tasks=[(c,t,k,1) for c in 'ABCD' for t in [4,8,12,16] for k in ['coarse_history','rich_recent','rich_history']]+[(c,t,'rich_history',2) for c in 'ABCD' for t in [4,8,12,16]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(case,tasks))
 print('64 PREDICTIONS FROZEN',flush=True)
