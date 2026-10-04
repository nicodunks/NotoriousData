import csv,json,statistics,collections
from pathlib import Path
O=Path.cwd()/'outputs/experiment';rows=list(csv.DictReader((O/'case_scores.csv').open()));rr={(x['clip'],int(x['cutoff']),x['target'],x['condition'],int(x['repeat'])):x for x in rows if x['scored']=='True'}
pairs=[]
for better,worse in [('reviewed_pose_history','reviewed_history'),('rich_history','coarse_history'),('rich_history','rich_recent'),('reviewed_history','rich_history'),('rich_history','persistence'),('rich_history','cross_bout_prevalence')]:
 for c in 'ABCD':
  for t in [4,8,12,16]:
   for target in ['phase','first_action','strike_attempt','takedown_attempt','ground_position_change']:
    a=rr.get((c,t,target,better,1));b=rr.get((c,t,target,worse,1))
    if a and b:pairs.append(dict(clip=c,cutoff=t,target=target,comparison=f'{better} minus {worse}',brier_delta=float(a['brier'])-float(b['brier']),accuracy_delta=int(a['correct']=='True')-int(b['correct']=='True')))
summary=[];groups={'A':'bout1','B':'bout2','C':'bout3','D':'bout3'}
for comp in sorted(set(x['comparison'] for x in pairs)):
 for target in ['phase','first_action','all_events']:
  a=[x for x in pairs if x['comparison']==comp and (x['target'] in ['strike_attempt','takedown_attempt','ground_position_change'] if target=='all_events' else x['target']==target)]
  if not a:continue
  summary.append(dict(comparison=comp,target=target,n=len(a),mean_brier_delta=statistics.mean(x['brier_delta'] for x in a),mean_accuracy_delta=statistics.mean(x['accuracy_delta'] for x in a),leave_one_bout_out=[dict(excluded=g,n=len(z),mean_brier_delta=statistics.mean(x['brier_delta'] for x in z),mean_accuracy_delta=statistics.mean(x['accuracy_delta'] for x in z)) for g in ['bout1','bout2','bout3'] if (z:=[x for x in a if groups[x['clip']]!=g])]))
with (O/'paired_deltas.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=pairs[0].keys());w.writeheader();w.writerows(pairs)
(O/'paired_summary.json').write_text(json.dumps(summary,indent=2))
repeat=[]
for c in 'ABCD':
 for t in [4,8,12,16]:
  p=[json.loads((O/'predictions'/f'predict_{c}_{t:02d}_rich_history_r{r}.json').read_text()) for r in [1,2]]
  repeat.append(dict(clip=c,cutoff=t,phase_top1_same=max(p[0]['phase_probs'],key=p[0]['phase_probs'].get)==max(p[1]['phase_probs'],key=p[1]['phase_probs'].get),action_top1_same=max(p[0]['next_action_probs'],key=p[0]['next_action_probs'].get)==max(p[1]['next_action_probs'],key=p[1]['next_action_probs'].get),mean_event_probability_difference=statistics.mean(abs(p[0]['event_probs'][k]-p[1]['event_probs'][k]) for k in p[0]['event_probs'])))
(O/'repeat_stability.json').write_text(json.dumps(repeat,indent=2));print('Paired comparisons and repeat stability saved.')
