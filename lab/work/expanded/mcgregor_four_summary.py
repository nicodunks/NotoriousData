import json,csv,math,numpy as np
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded/mcgregor/four_bout';O.mkdir(exist_ok=True);manifest=[]
for folder in ['pilot','expansion','expansion_extra']:
 base=R/'outputs/expanded/mcgregor'/folder
 if (base/'manifest.json').exists():
  manifest += [{**r,'base_dir':str(base)} for r in json.load(open(base/'manifest.json'))]
rows=[];metrics=[];gates=[]
def within(t,interval):return interval['start']<=t<interval['end']
def length(a,b):return float(np.linalg.norm((np.array(a)-b)*[640,360]))
for row in manifest:
 cid=row['anonymous_id'];base=Path(row['base_dir']);p=base/'labels'/f'{cid}.json';mp=base/'audit'/f'{cid}.metadata.json';pp=base/'audit'/f'{cid}.prompt.txt'
 if not all(q.exists() for q in [p,mp,pp]):continue
 if 'Validate the supplied interval-specific shorts and wrist-tape combination' not in pp.read_text():continue
 meta=json.load(open(mp));assert not meta['tools_used'];d=json.load(open(p));identity=d['target_identity_visibility'];identity='unresolved_anchor' if cid=='X06' else identity
 live=d['live_fight']=='yes' and d['likely_replay']=='no';standing=[r for r in d['phases'] if r['phase']=='standing'];stand=sum(max(0,min(7.75,r['end'])-max(0,r['start'])) for r in standing) if live else 0;stand=min(7.75,stand);identifiable=identity=='high';observable=live and identifiable and stand>0
 def stand_event(r):return observable and any(r['start']>=s['start'] and r['end']<=s['end'] for s in standing)
 counts={k:sum(1 for e in d['strike_attempts'] if stand_event(e) and e['type']==k and e['certainty'] in ('high','medium')) if observable else None for k in ('punch','kick','knee','elbow','spinning','unclear')}
 feet=min(stand,d['camera']['feet_observable_seconds_estimate']) if observable else 0;quiet=sum(max(0,min(q['end'],s['end'],7.75)-max(q['start'],s['start'],0)) for q in d['usable_quiet_standing_intervals'] for s in standing) if observable else 0
 fw=[e for e in d['footwork'] if stand_event(e) and e['certainty'] in ('high','medium')];bounce=sum(e.get('count_lower_bound',0) for e in fw if e['pattern']=='bounce_cycle') if feet>0 and fw else None
 rows.append({'anonymous_id':cid,'source':row['source'],'source_start':row['start'],'observed_span_seconds':7.75,'identity_status':identity,'live_fight':d['live_fight'],'likely_replay':d['likely_replay'],'standing_seconds_estimate':stand,'identifiable_standing_seconds_estimate':stand if identifiable else 0,'feet_visible_seconds_estimate':feet,'quiet_standing_seconds_estimate':quiet,'quiet_and_feet_exposure_upper_bound':min(feet,quiet),'camera_view':d['camera']['view'],'camera_motion_confound':d['camera']['camera_motion_confound'],'bounce_cycles_reported_lower_bound':bounce,**{k+'_attempt_reported_lower_bound':v for k,v in counts.items()}})
 for a in d['point_anchors']:
  t=a['time'];pts=a['points_normalized'];owned=set(a['visible_owned_points']);reasons=[]
  if not live:reasons.append('not verified live nonreplay fight')
  if not identifiable:reasons.append('target identity unresolved/low visibility')
  if not any(within(t,s) for s in standing):reasons.append('anchor not inside standing phase')
  if not any(within(t,q) for q in d['usable_quiet_standing_intervals']):reasons.append('anchor not inside quiet interval (half-open boundaries)')
  if not a['eligible_full_body_quiet_stance']:reasons.append('annotator full-body quiet stance gate false')
  if feet<=0:reasons.append('no independently visible feet exposure')
  required=['left_ankle','right_ankle','left_hip','right_hip']
  if not all(k in owned and pts.get(k) is not None for k in required):reasons.append('paired ankles/hips missing or ownership unverified')
  gates.append({'anonymous_id':cid,'time':t,'accepted':not reasons,'reasons':reasons})
  if reasons:continue
  u=max(.02,a.get('point_coordinate_uncertainty_normalized',.02));error=2*math.hypot(640*u,360*u);span=length(pts['left_ankle'],pts['right_ankle']);leg=(length(pts['left_hip'],pts['left_ankle'])+length(pts['right_hip'],pts['right_ankle']))/2
  if leg<=error:continue
  metrics.append({'anonymous_id':cid,'source':row['source'],'source_time':row['start']+t,'local_time':t,'metric':'projected ankle separation / mean projected hip-to-ankle distance','value':span/leg,'coordinate_uncertainty_normalized':u,'conservative_coordinate_error_range':[max(0,span-error)/(leg+error),(span+error)/(leg-error)],'provenance':'blinded approximate image coordinates; not calibrated confidence interval or gold truth'})
with (O/'window_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(O/'image_metrics.json').write_text(json.dumps(metrics,indent=2));(O/'anchor_eligibility.json').write_text(json.dumps(gates,indent=2));bouts=[]
for sid in ['obgm6JNtyVo','BCOy-PG8EIw','khabib-mcgregor','6yu2AWK4rxo']:
 rs=[r for r in rows if r['source']==sid];bouts.append({'source':sid,'bout_units':1,'windows_labeled':len(rs),'windows_registered':sum(r['source']==sid for r in manifest),'observed_span_seconds':len(rs)*7.75,'identifiable_standing_seconds_estimate':sum(r['identifiable_standing_seconds_estimate'] for r in rs),'feet_visible_seconds_estimate':sum(r['feet_visible_seconds_estimate'] for r in rs),'quiet_standing_seconds_estimate':sum(r['quiet_standing_seconds_estimate'] for r in rs),'quiet_and_feet_exposure_upper_bound':sum(r['quiet_and_feet_exposure_upper_bound'] for r in rs),'attempt_reported_lower_bounds':{k:sum(r[k+'_attempt_reported_lower_bound'] for r in rs if r[k+'_attempt_reported_lower_bound'] is not None) for k in ('punch','kick','knee','elbow','spinning','unclear')},'bounce_cycles_reported_lower_bound':sum(r['bounce_cycles_reported_lower_bound'] for r in rs if r['bounce_cycles_reported_lower_bound'] is not None),'bounce_observable_windows':sum(r['bounce_cycles_reported_lower_bound'] is not None for r in rs),'accepted_static_metric_anchors':sum(m['source']==sid for m in metrics)})
result={'status':'complete' if len(rows)==28 else 'incomplete','bout_units':4,'windows':len(rows),'bouts':bouts,'aggregate_guard_stance_comparison':'WITHHELD pending view/projection matching; per-anchor projected values retained descriptively only. Original systematic control has no eligible late static anchor.','zero_semantics':'Attempt totals are reported visible lower bounds over identifiable live standing samples. Missing/unresolved exposures are null, never zero. Empty footwork arrays with no observable footwork are null.','limits':['Four selected bouts/opponents; cannot establish career trajectory or causal style change.','Live standing exposures and views uneven; source-time sampling retains ground/nonfight exclusions.','4Hz misses rapid attempts and cycles; no contact/damage truth.','2D geometry depends on camera, body orientation, distance; no 3D stance or body weight inference.','Names/faces/clothing remain readable; incomplete era blinding.','Adjacent frames/windows nested within bouts; no independent frame N.','X06 absent annotation reclassified unresolved_anchor, not physical absence.']};(O/'bout_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
lines=['Four-bout evidence study: exploratory conditional standing supplement plus original systematic controls.','Label roster includes original pilot plus targeted live-standing windows from two additional bout contexts. Full pixels and per-bout clothing anchors retained.','']
for b in bouts:lines.append(f"{b['source']}: {b['identifiable_standing_seconds_estimate']:.2f}s identifiable standing, {b['feet_visible_seconds_estimate']:.2f}s feet visible estimate, {b['quiet_standing_seconds_estimate']:.2f}s quiet standing estimate; reported attempt lower bounds {b['attempt_reported_lower_bounds']}; bounce-cycle lower bound {b['bounce_cycles_reported_lower_bound']} in {b['bounce_observable_windows']} observable windows.")
lines+=['','Aggregate projected-stance comparison withheld until camera/view matching is established. Per-anchor values and broad coordinate-error ranges are descriptive only. The original systematic controls have no eligible late static anchor; adaptive quiet supplement handled separately.','Observed patterns are steps, shuffles, withdrawal, guard changes and attacks in specific opponent contexts. These data do not establish a career style shift.','','Limitations: '+ '; '.join(result['limits']),'','Exact gates: anchor_eligibility.json. Per-window roster: window_summary.csv. Each raw label retains timed evidence and uncertainty.']
(O/'findings.txt').write_text('\n'.join(lines)+'\n')
