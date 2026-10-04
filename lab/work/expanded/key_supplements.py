import json,sys,concurrent.futures,hashlib
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as b
P=R/'outputs/expanded/key_moments';O=P/'supplements';O.mkdir(exist_ok=True);b.O=O
for d in ['audit','past_event_presence','matched_pose_predictions']:(O/d).mkdir(exist_ok=True)
protocol=json.load(open(P/'protocol.json'));DEFS=protocol['definitions'];TARGETS=protocol['targets'];CASES=protocol['cases']

def pastflag(case):
 c,t=case;hist=json.load(open(P/'prefixes'/f'{c}_{t}.json'));events=[]
 for h in hist:
  for e in h.get('events',[]):
   st=e.get('start',e.get('time_start'));en=e.get('end',e.get('time_end',st))
   if st is None:continue
   if st>=t-4 and st<t:events.append(e)
 prompt='Classify ONLY actuallyobserved pastkeyevents in ['+str(t-4)+','+str(t)+') fromsuppliedCAUSALannotatedevents. No prediction, noimages, tools, future, names, fightknowledge. '+DEFS+'\nIfa type isnot supported bytheannotation distinguishno fromunclear; probeorhandfightingisnotautomaticallystrike. Eventsnotlistedmayhave beenunobservable; visiblelastphase and recordeduncertaintieshelpassessnegativeconfidence. Return JSON {past_interval:['+str(t-4)+','+str(t)+'],targets:{'+','.join(k+':{occurrence:yes|no|unclear,evidence}' for k in TARGETS)+'},limits}.\n'+json.dumps({'events_with_onsets_in_interval':events,'last_phase':hist[-1]['end_phase'],'recorded_visibility':[x.get('visibility') for x in hist if x['end']>t-4],'recorded_uncertainties':[x.get('uncertainties') for x in hist if x['end']>t-4]})
 a=b.call(f'{c}_pastflag_{t}',prompt);(O/'past_event_presence'/f'{c}_{t}.json').write_text(json.dumps(a,indent=2));print('PASTFLAG',c,t,flush=True)

def matched(t):
 hist=json.load(open(P/'prefixes'/f'B_{t}.json'));body=json.load(open(R/'outputs/experiment/body_data/B_rtmw_133.json'));rows=body['frames'];frames=[];js=list(range(23))+[91,95,99,103,107,111,112,116,120,124,128,132]
 for when in [t-4+i*.5 for i in range(8)]:
  r=min((r for r in rows if r['time']<t),key=lambda r:abs(r['time']-when));frames.append({'time':r['time'],'people':[{'box':z['box'],'keypoints_indices':js,'keypoints':[z['keypoints'][j] for j in js],'ownership':'unverified detection, notstablefighteridentity'} for z in r['people'][:4]]})
 pose={'width':body['width'],'height':body['height'],'raw_scores_not_probabilities':True,'frames':frames};assert all(r['time']<t for r in frames)
 prompt='Predict upcoming4seconds keyeventoccurrence ONLY from suppliedpaststructuredobservations. No tools,names,fightmemories,files,web,futurefactsorrawimages. '+DEFS+'\nFuturewindow ['+str(t)+','+str(t+4)+'). Independent Bernoulli probabilities0–1; no sum1 requirement. Return JSON {probabilities:{'+','.join(k+':probability' for k in TARGETS)+'},rationale_by_target:{'+','.join(k+':reason' for k in TARGETS)+'},uncertainties}.\n'+json.dumps({'past_observations':hist,'past_pose':pose})
 a=b.call(f'B_matchedpose_{t}',prompt);(O/'matched_pose_predictions'/f'B_{t}.json').write_text(json.dumps(a,indent=2));print('MATCHEDPOSE',t,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(pastflag,CASES))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(matched,[4,6,7]))
rows=[]
for c,t in CASES:
 ref=json.load(open(P/'references'/f'{c}_{t}.json'));past=json.load(open(O/'past_event_presence'/f'{c}_{t}.json'))
 for k in TARGETS:
  tr=ref['targets'][k]['occurrence'];seen=past['targets'][k]['occurrence'];p=1 if seen=='yes' else 0 if seen=='no' else None
  rows.append({'clip':c,'cutoff':t,'target':k,'condition':'recent_event_persistence','past_occurrence':seen,'truth':tr,'probability':p,'brier':(p-(tr=='yes'))**2 if p is not None and tr!='unclear' else None,'abstained':p is None})
 for k in TARGETS:
  if c=='B':
   pr=json.load(open(O/'matched_pose_predictions'/f'B_{t}.json'))['probabilities'][k];tr=ref['targets'][k]['occurrence'];rows.append({'clip':c,'cutoff':t,'target':k,'condition':'rich_pose_matched_history','truth':tr,'probability':pr,'brier':(pr-(tr=='yes'))**2 if tr!='unclear' else None})
(O/'case_scores.json').write_text(json.dumps(rows,indent=2));(O/'protocol.json').write_text(json.dumps({'past_event_presence':'FreshSolclassification ofonlycausalannotatedeventonsets last4s; unobservableorambiguous mayabstain. Predictrepetition1ifpastyes,0ifpastno,abstainunknown; excludesongoingactionwithoutnewonset. Noimages/futureinputs.','B_matched_pose':'Supplementarycorrectedadapter samples2Hzlast4s35pointindices matchingE-Hconfiguration. InitialB adapterlast8native12Hzrows retainedseparately, notsilentlyreplaced. Supplementalslightlydifferentpromptwordingnotpureadapterablation.','training':False,'model':'gpt-6.1-sol'},indent=2));print('SCORED',flush=True)
