import json, csv, math, collections, statistics
from pathlib import Path
ROOT=Path.cwd();O=ROOT/'outputs/experiment';PH=['standing','clinch','ground','unclear'];AC=['punch','kick_knee_elbow','takedown_entry','ground_strike','ground_control_change','none','unclear'];EV=['strike_attempt','takedown_attempt','ground_position_change'];GROUP={'A':'bout1','B':'bout2','C':'bout3','D':'bout3'}
def read(p):return json.loads(p.read_text())
refs={}
for c in 'ABCD':
 for t in range(0,20,4):
  p=O/'adjudication'/f'{c}_{t:02d}.json'
  refs[c,t]=read(p if p.exists() else O/'verification'/f'{c}_{t:02d}.json')
rows=[];preds=[]
for p in sorted((O/'predictions').glob('*.json')):
 x=read(p);c=x['clip'];t=x['cutoff'];r=refs[c,t];past=[z for z in read(O/'labels'/f'{c}.json') if z['end']<=t]
 assert past and past[-1]['end']==t
 preds.append(x)
 def row(target,truth,probs,classes,known):
  
  if target not in EV:
   total=sum(probs.values());probs={k:v/total for k,v in probs.items()}
  top=max(probs,key=probs.get);brier=((probs.get('yes',0)-float(truth=='yes'))**2 if target in EV else sum((probs.get(k,0)-(1 if k==truth else 0))**2 for k in classes)) if known else None
  rows.append(dict(clip=c,cutoff=t,condition=x['condition'],repeat=x['repeat'],target=target,reference=truth,scored=known,predicted=top,correct=(top==truth) if known else None,brier=brier,probability_of_reference=probs.get(truth) if known else None,probabilities=json.dumps(probs)))
 row('phase',r['end_phase'],x['phase_probs'],PH,r['end_phase']!='unclear')
 row('first_action',r.get('first_action','unclear'),x['next_action_probs'],AC,r.get('first_action','unclear')!='unclear')
 for e in EV:
  val=r['event_flags'][e];pv=x['event_probs'][e];truth='yes' if val=='yes' else 'no'
  row(e,truth,{'yes':pv,'no':1-pv},['yes','no'],val!='unclear')
# Baselines frozen using only the last past causal annotation, or cross-bout empirical prevalence.
for c in 'ABCD':
 for t in [4,8,12,16]:
  r=refs[c,t];past=[z for z in read(O/'labels'/f'{c}.json') if z['end']<=t][-1]
  others=[refs[cc,tt] for cc in 'ABCD' if GROUP[cc]!=GROUP[c] for tt in [4,8,12,16]]
  for cond in ['persistence','cross_bout_prevalence']:
   targets={'phase':(r['end_phase'],PH,r['end_phase']!='unclear'),'first_action':(r.get('first_action','unclear'),AC,r.get('first_action','unclear')!='unclear')}
   for e in EV:targets[e]=(r['event_flags'][e],['yes','no'],r['event_flags'][e]!='unclear')
   for target,(truth,classes,known) in targets.items():
    if cond=='persistence':
     if target=='phase':probs={k:float(k==past['end_phase']) for k in classes}
     elif target=='first_action':continue
     else:
      last=past['event_flags'][target];pv=1 if last=='yes' else 0 if last=='no' else .5;probs={'yes':pv,'no':1-pv}
    else:
     vals=[z['end_phase'] if target=='phase' else z.get('first_action','unclear') if target=='first_action' else z['event_flags'][target] for z in others]
     vals=[v for v in vals if v!='unclear'];valid=[k for k in classes if k!='unclear'];counts=collections.Counter(vals);den=len(vals)+len(valid);probs={k:(counts[k]+1)/den if k in valid else 0 for k in classes}
    top=max(probs,key=probs.get);bs=((probs.get('yes',0)-float(truth=='yes'))**2 if target in EV else sum((probs[k]-float(k==truth))**2 for k in classes)) if known else None
    rows.append(dict(clip=c,cutoff=t,condition=cond,repeat=1,target=target,reference=truth,scored=known,predicted=top,correct=(top==truth) if known else None,brier=bs,probability_of_reference=probs.get(truth) if known else None,probabilities=json.dumps(probs)))
summary=[]
for cond,rep in sorted(set((x['condition'],x['repeat']) for x in rows)):
 for target in ['phase','first_action']+EV+['all_events']:
  relevant=[x for x in rows if x['condition']==cond and x['repeat']==rep and (x['target'] in EV if target=='all_events' else x['target']==target)];valid=[x for x in relevant if x['scored']]
  if not relevant:continue
  tp=sum(x['reference']=='yes' and x['predicted']=='yes' for x in valid);fp=sum(x['reference']=='no' and x['predicted']=='yes' for x in valid);fn=sum(x['reference']=='yes' and x['predicted']=='no' for x in valid)
  summary.append(dict(condition=cond,repeat=rep,target=target,n=len(valid),possible=len(relevant),accuracy=sum(x['correct'] for x in valid)/len(valid) if valid else None,brier=statistics.mean(x['brier'] for x in valid) if valid else None,positive_support=sum(x['reference']=='yes' for x in valid),precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,reference_counts=dict(collections.Counter(x['reference'] for x in valid)),per_clip={c:dict(n=len(z),accuracy=statistics.mean(v['correct'] for v in z),brier=statistics.mean(v['brier'] for v in z)) for c in 'ABCD' if (z:=[v for v in valid if v['clip']==c])}))
with (O/'case_scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(O/'scores.json').write_text(json.dumps(summary,indent=2))
agreements=[]
for c in 'ABCD':
 for z in read(O/'labels'/f'{c}.json'):
  v=read(O/'verification'/f'{c}_{z["start"]:02d}.json')
  agreements.append(dict(clip=c,start=z['start'],phase_agree=z['end_phase']==v['end_phase'],event_agreement={e:z['event_flags'][e]==v['event_flags'][e] for e in EV},causal=z['event_flags'],review=v['event_flags']))
(O/'label_agreement.json').write_text(json.dumps(agreements,indent=2))
for s in summary:
 if s['target'] in ['phase','first_action','all_events']:print(s['condition'],s['repeat'],s['target'],s['n'],s['accuracy'],s['brier'])
