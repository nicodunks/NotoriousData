import json,sys,cv2,numpy as np,hashlib,concurrent.futures,time
from pathlib import Path
from PIL import Image,ImageDraw
R=Path.cwd();sys.path.insert(0,str(R/'work/expanded'));import causal_v2 as b
O=R/'outputs/expanded/key_moments/dense_prefix';O.mkdir(exist_ok=True);b.O=O
for x in ['audit','evidence','observations','predictions']:(O/x).mkdir(exist_ok=True)
CASES=[(c,t) for c in ['B','K'] for t in [4,6,7]];TARGETS=['strike_attempt','takedown_attempt','submission_entry','major_control_transition']
DEFS=json.load(open(R/'outputs/expanded/key_moments/protocol.json'))['definitions']
P={'model':'gpt-6.1-sol','reasoning':'high','training':False,'cases':CASES,'selection':'Two previouslyobservedfailureepisodes: B takedown andK reversal. Retrospective selected stress test, notheldout. Dependent3cutoffs each.','hypothesis':'Densecausalprecutoffimage reviews +explicit state/movement schema mayretain observablestatechanges absent sparseprose. Annotationformat, sampling, andtwo-reviewerbundle change together; notisolatedframerateablation.','causality':'Last4seconds strictlybeforecutoff12Hznativefullframes; bothannotationreviewers seeONLYsamepastimages, noanswers/futureeventrefs/predictions. Predictor textonlybothreviews+oldercausalhistory. No fightnames/intents/futurememory.','reference':'Existingfrozenfutureeventreferences, separatelaternativereview sensitivity. Futuredata neverobservations.','model_consensus':'Both independentreview texts passed; unresolveddisagreementnotvotedintotruth. SameSolmodelcorrelatederrors remain.','created_unix':time.time()};(O/'protocol.json').write_text(json.dumps(P,indent=2))

def evidence(c,t):
 path=R/'outputs/experiment/clips/B.mp4' if c=='B' else R/'outputs/expanded/key_moments/control_transition/K.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4));frames=[];manifest=[]
 for i in range(48):
  requested=t-4+i/12;ix=int(round(requested*fps));actual=ix/fps
  if actual>=t:ix-=1;actual=ix/fps
  assert actual<t and actual>=t-4-.04
  cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,f=cap.read();assert ok;frames.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)));manifest.append({'requested':requested,'native_frame':ix,'actual':actual,'causal':actual<t})
 cap.release();paths=[]
 for st in range(0,48,4):
  im=Image.new('RGB',(w*2,(h+32)*2),'#10171c');d=ImageDraw.Draw(im)
  for j,frame in enumerate(frames[st:st+4]):
   x=j%2*w;y=j//2*(h+32);im.paste(frame,(x,y));d.text((x+10,y+h+8),f'past t={manifest[st+j]["actual"]:.4f}s',fill='white')
  p=O/'evidence'/f'{c}_{t}_{st//4:02d}.jpg';im.save(p,quality=92);paths.append(p)
 (O/'evidence'/f'{c}_{t}_manifest.json').write_text(json.dumps({'source':str(path.relative_to(R)),'fps':fps,'width':w,'height':h,'samples':manifest,'all_before_cutoff':True},indent=2));return paths
IM={case:evidence(*case) for case in CASES}

def observe(job):
 c,t,reviewer=job
 prompt='Use ONLY supplied orderedPASTnativeimage sequences throughbutstrictlyBEFORE '+str(t)+'s. No tools, files, web, names, knownfightmemory, intentprediction, futureeventspeculation. Review last4seconds['+str(t-4)+','+str(t)+'). Nativefullframepages mayberesizedbyfrontend; donotinvent precision oroccludedjoints. ClothingcolorIDs only; neverAlpha/Beta. Describe ONLY visiblefactualstateandmotion, notwhatwillhappen. Return JSON {interval:[start,end],fighters:[{color_id,clothing,ownership_limits}],end_phase,visible_distance_and_orientation,guard_and_hand_position,head_torso_level_changes,feet_knees_base_changes,grips_and_supports,recent_committed_action_onsets:[{time_interval,actor,action,evidence,certainty}],time_sliced_observations:[{start,end,actor,visible_state_change,evidence_timestamps}],occlusions_and_unknowns}. DoNOTcalla loweringorstepa takedownpreparation withoutactualcommittedentry. DoNOTcallhiprollasweepbeforeestablishedtransition. No physicalforce/damage/hidden anatomy. Give short precise observations withtimes andactorambiguity.'
 a=b.call(f'{c}_{t}_past_reviewer_{reviewer}',prompt,IM[(c,t)]);(O/'observations'/f'{c}_{t}_{reviewer}.json').write_text(json.dumps(a,indent=2));print('OBS',c,t,reviewer,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(observe,[(c,t,r) for c,t in CASES for r in [1,2]]))

def predict(case):
 c,t=case
 hist=json.load(open(R/'outputs/expanded/key_moments/prefixes'/f'{c}_{t}.json')) if c=='B' else [x for x in json.load(open(R/'outputs/expanded/key_moments/control_transition/observations/K.json')) if x['end']<=t]
 assert all(x['end']<=t for x in hist)
 a1=json.load(open(O/'observations'/f'{c}_{t}_1.json'));a2=json.load(open(O/'observations'/f'{c}_{t}_2.json'))
 prompt='Forecast anyNEWkeyeventonset in next4seconds['+str(t)+','+str(t+4)+') usingONLYcausalpasttext. No images,names,fightmemory,tools,files,web,futurefacts. '+DEFS+'\nTwo independentpastimage reviews maydisagree; preserveambiguity. Bothfromsamevisionmodel, agreementnotproof. Return JSON {probabilities:{'+','.join(k+':independentBernoulli0to1' for k in TARGETS)+'},rationale_by_target:{'+','.join(k+':reason' for k in TARGETS)+'},uncertainties}. ProbabilitiesneedNOTsum1. Pastdata:\n'+json.dumps({'earlier_causal_history':hist,'dense_past_review_1':a1,'dense_past_review_2':a2})
 a=b.call(f'{c}_{t}_dense_predict',prompt);(O/'predictions'/f'{c}_{t}.json').write_text(json.dumps(a,indent=2));print('PRED',c,t,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(predict,CASES))
rows=[]
for c,t in CASES:
 refpath=R/'outputs/expanded/key_moments/references'/f'{c}_{t}.json' if c=='B' else R/'outputs/expanded/key_moments/control_transition/references'/f'{c}_{t}.json';ref=json.load(open(refpath));pred=json.load(open(O/'predictions'/f'{c}_{t}.json'))
 for k in TARGETS:
  p=pred['probabilities'][k];assert 0<=p<=1;tr=ref['targets'][k]['occurrence'];rows.append({'clip':c,'cutoff':t,'target':k,'truth':tr,'probability':p,'brier':(p-(tr=='yes'))**2 if tr!='unclear' else None,'alert':p>=.5,'onsets_seconds':ref['targets'][k]['onsets_seconds'],'condition':'dense_two_review_history'})
(O/'case_scores.json').write_text(json.dumps(rows,indent=2));print('SCORED',flush=True)
