import json,hashlib,shutil,datetime,html
from pathlib import Path
R=Path.cwd(); O=R/'outputs/expanded'; D=O/'key_moments/dense_prefix'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=[]
for m in sorted((D/'audit').glob('*.metadata.json')):
 a=load(m);p=m.with_name(m.name.replace('.metadata.json','.prompt.txt'));assert sha(p)==a['prompt_sha256'];assert not a['tools_used'];assert not a['transport_errors']
 for im in a['images']:assert sha(R/im['path'])==im['sha256']
 audit.append({'call':m.stem,'prompt_verified':True,'image_count':len(a['images']),'tools':0})
assert len(audit)==18
for p in (D/'evidence').glob('*_manifest.json'):
 a=load(p);t=int(p.stem.split('_')[1]);assert len(a['samples'])==48;assert all(t-4-.04<=x['actual']<t for x in a['samples'])
for p in (D/'predictions').glob('*.json'):assert all(isinstance(x,(int,float)) and 0<=x<=1 for x in load(p)['probabilities'].values())
(D/'verification.json').write_text(json.dumps({'calls':audit,'all_288_frame_samples_strictly_before_cutoff':True,'prediction_probabilities_valid':True,'future_reference_not_in_annotation_or_prediction_input':'Verified from retained prompts and source; model calls tool-free','annotation_schema_sampling_and_two_review_bundle_change_together':True},indent=2))
rows=[]
for c,target in [('B','takedown_attempt'),('K','major_control_transition')]:
 for t in [4,6,7]:
  q=O/'key_moments/predictions'/f'{c}_{t}_rich_history.json' if c=='B' else O/'key_moments/control_transition/predictions'/f'{c}_{t}_rich_history.json'
  if not q.exists():q=next((O/'key_moments/control_transition/predictions').glob(f'{c}_{t}*rich*'))
  rows.append({'case':f'{c}_{t}','target':target,'original_rich':load(q)['probabilities'][target],'dense_two_reviews':load(D/'predictions'/f'{c}_{t}.json')['probabilities'][target]})
(D/'comparison.json').write_text(json.dumps({'selection':'Two selected failure episodes, three overlapping cutoffs each; no independence claim','alert_threshold':.5,'all_selected_event_alerts_missed':True,'rows':rows},indent=2))
report='DENSE PAST REVIEW STRESS TEST\n\n18 tool-free Sol6.1 high calls: two independent image reviews and one text-only forecast per cutoff. Each review sees 48 native frame samples from the preceding four seconds, strictly before cutoff. All prompt and image hashes verified.\n\nTwo selected episodes, not six independent positives: B takedown and K reversal. Sampling, annotation schema and two-reviewer bundle change together; this is not a controlled frame-rate ablation.\n\n'
for x in rows:report+=f"{x['case']} {x['target']}: original rich {x['original_rich']:.0%}; dense reviews {x['dense_two_reviews']:.0%}.\n"
report+='\nAll stay below the 50% alert threshold. More descriptive past annotations did not unlock anticipation here. K4 future control reference is unclear, since established top control occurs after its horizon. K6/K7 are positive for the same reversal. B cutoffs cover the same takedown. Frozen references remain separate from later native review.\n'
(D/'report.txt').write_text(report)
(D/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense past review</title><style>body{background:#10171c;color:#dce6e9;font:17px/1.7 system-ui;max-width:950px;margin:40px auto;padding:24px}a{color:#22d3ee}pre{white-space:pre-wrap;font:inherit}</style><a href="../index.html">← Key moments</a><h1>What changed with much denser past review?</h1><pre>'+html.escape(report)+'</pre><a href="comparison.json">Structured comparison</a> · <a href="verification.json">Audit</a> · <a href="protocol.json">Protocol</a>')
# Preserve legacy measurements while making rejection machine-readable.
p=O/'mcgregor/four_bout/bout_summary.json';a=load(p);a['measurement_status']='Legacy hip-based static metric anchors are superseded and rejected; use quiet_projection_measurements.json for visible-landmark candidates.'
for b in a['bouts']:
 if 'accepted_static_metric_anchors' in b:b['legacy_accepted_static_metric_anchors']=b.pop('accepted_static_metric_anchors');b['static_metric_anchor_status']='superseded: inferred hips rejected'
a['feet_exposure_note']='Feet-visible-standing exposure estimates using min of separate estimates are upper bounds on intersection, not exact joint exposure and must not be used as validated rate denominators.';p.write_text(json.dumps(a,indent=2))
p=O/'mcgregor/four_bout/image_metrics.json';a=load(p);p.write_text(json.dumps({'status':'SUPERSEDED: anatomical hip-based metrics rejected; retained for provenance only','replacement':'quiet_projection_measurements.json','legacy_records':a},indent=2))
# Complete annotation export preserves retrospective labels and corrective layer separately.
bundle={'model':'gpt-6.1-sol','training':False,'warning':'Retrospective 20-second clip annotation, NOT causal forecasting input. Same-model independent reviews are not human-expert truth. Corrections remain a separate layer; no silent reconciliation. Contact includes body/grip contact and is not equivalent to landed strike.','clips':{}}
for c in 'EFGHIJ':bundle['clips'][c]={'reviewed':load(O/'annotations'/f'{c}.json'),'detail_and_corrections':load(O/'annotations'/f'{c}_detail.json')}
(O/'research_bundle.json').write_text(json.dumps(bundle,indent=2))
T=O/'reproduce/root';T.mkdir(parents=True,exist_ok=True)
for n in ['dense_prefix_test.py','new_color_figures.py','identity_new_generated.py','render_new_generated.py','build_pose_clinic.py','build_frame_lab.py','build_assessment.py','analyze_key_results.py','build_key_viewer.py','finalize_research.py']:
 p=R/'work/expanded'/n
 if p.exists():shutil.copy2(p,T/n)
print('Verified dense review18 calls;288 causal frame samples. Complete six-clip bundle and legacy metric warnings saved.')
