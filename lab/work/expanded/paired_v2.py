import json,collections
from pathlib import Path
O=Path('outputs/expanded/causal_v2');p=O/'case_scores.json'
if not p.exists():print('Full study not scored yet');raise SystemExit(0)
rows=json.load(open(p));paired=[]
for r in rows:
 if r['condition'] not in ['rich_history','rich_pose_history'] or not r['eligible']:continue
 base=next(x for x in rows if x['clip']==r['clip'] and x['cutoff']==r['cutoff'] and x['target']==r['target'] and x['condition']=='coarse_history')
 paired.append({'clip':r['clip'],'bout_group':r['bout_group'],'cutoff':r['cutoff'],'target':r['target'],'condition':r['condition'],'comparator':'coarse_history','brier_delta':r['brier']-base['brier'],'negative_means_lower_error':True})
by=[]
for group in sorted({r['bout_group'] for r in rows}):
 for cond in sorted({r['condition'] for r in rows}):
  for target in ['phase','action']:
   rs=[r for r in rows if r['bout_group']==group and r['condition']==cond and r['target']==target and r['eligible']]
   if rs:by.append({'bout_group':group,'condition':cond,'target':target,'n_cases':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'mean_brier':sum(r['brier'] for r in rs)/len(rs)})
(O/'paired_deltas.json').write_text(json.dumps(paired,indent=2));(O/'bout_scores.json').write_text(json.dumps(by,indent=2));(O/'analysis_limits.json').write_text(json.dumps({'bout_units':3,'windows_and_clips_nested_in_bouts':True,'confidence_intervals_or_significance_claims':False,'reference':'Independent model-reviewed, not expert ground truth','sampling':'Selected ordinary clips; no population generalization'},indent=2));print('Paired and bout-group summaries complete')
