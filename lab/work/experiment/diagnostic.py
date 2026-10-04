from run import *
from colors import colorize
from predict import SCHEMA
from concurrent.futures import ThreadPoolExecutor
(OUT/'diagnostic').mkdir(exist_ok=True)
R=RULES.replace('Images are timestamped 8 Hz','Images are timestamped at native source frame rate (about 30 Hz)')
T=LABEL_TEMPLATE.replace('Alpha|Beta|unclear','Green|Black/gray|unclear').replace('"Alpha":"distinctive clothing","Beta":"distinctive clothing"','"Green":"green shorts","Black/gray":"black shorts"')
past0=colorize(json.loads((OUT/'labels/B.json').read_text())[0])
def label(st,en,kind):
 prompt=R+f'\nUse clothing colors Green and Black/gray, not generic pseudonyms. Label ONLY interval [{st},{en})s. '+('Earlier causal annotation: '+json.dumps(past0) if kind=='past' else 'Independent reference review: no forecasts supplied.')+'\n'+T+' Add first_action with category punch/kick_knee_elbow/takedown_entry/ground_strike/ground_control_change/none/unclear, first_action_time, and first_action_evidence. First observable committed action; sustained grips alone do not count; unresolved simultaneous ordering is unclear. End-state is the last sampled frame. Do not infer later events.'
 obj=call(f'diagnostic_{kind}_{st}_{en}',prompt,[BASE/'native/B'/f'{j:02d}.jpg' for j in range(st,en)]);obj.update(start=st,end=en);(OUT/'diagnostic'/f'{kind}_{st}_{en}.json').write_text(json.dumps(obj,indent=2));return obj
p46=label(4,6,'past');p67=label(6,7,'past')
def forecast(task):
 cutoff,cond=task;history=[past0,p46]+([p67] if cutoff==7 else [])
 if cond=='coarse_history':data=[{k:x[k] for k in ['start','end','start_phase','end_phase','visibility']} for x in history]
 elif cond=='rich_recent':data=history[-1:]
 else:data=history
 prompt='You are forecasting an anonymous MMA sequence. Use ONLY supplied annotations ending at or before cutoff. No tools, images, names, fight memories, web or future data. Predict end-phase and events ANYWHERE in the NEXT TWO SECONDS, plus the first committed action. These model-generated observations may be wrong. Give calibrated probabilities.\n'+f'Cutoff={cutoff}; horizon=2; condition={cond}.\nAnnotations:\n'+json.dumps(data)+'\n'+SCHEMA
 obj=call(f'diagnostic_predict_{cutoff}_{cond}',prompt);obj.update(cutoff=cutoff,horizon=2,condition=cond);(OUT/'diagnostic'/f'forecast_{cutoff}_{cond}.json').write_text(json.dumps(obj,indent=2));print('DIAGNOSTIC FORECAST',cutoff,cond,flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(forecast,[(t,c) for t in [6,7] for c in ['coarse_history','rich_recent','rich_history']]))
# Freeze forecasts before target review.
ref68=label(6,8,'reference');ref79=label(7,9,'reference')
rows=[]
for p in sorted((OUT/'diagnostic').glob('forecast*.json')):
 x=json.loads(p.read_text());ref=ref68 if x['cutoff']==6 else ref79
 rows.append(dict(cutoff=x['cutoff'],condition=x['condition'],p_takedown=x['event_probs']['takedown_attempt'],p_ground=x['phase_probs']['ground'],predicted_phase=max(x['phase_probs'],key=x['phase_probs'].get),reference_phase=ref['end_phase'],predicted_first_action=max(x['next_action_probs'],key=x['next_action_probs'].get),reference_first_action=ref['first_action'],reference_takedown=ref['event_flags']['takedown_attempt']))
(OUT/'diagnostic/results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
