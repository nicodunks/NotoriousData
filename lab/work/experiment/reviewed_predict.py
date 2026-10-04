from predict import SCHEMA
from run import *
from colors import colorize
def task(pair):
 c,t=pair;data=[]
 for st in range(0,t,4):
  p=OUT/'adjudication'/f'{c}_{st:02d}.json';p=p if p.exists() else OUT/'verification'/f'{c}_{st:02d}.json'
  x=colorize(json.loads(p.read_text()));data.append({k:v for k,v in x.items() if k not in ['adjudication_notes','first_action','first_action_time','first_action_evidence']})
 prompt='You are forecasting an anonymous MMA sequence. You must use ONLY the supplied annotation record available at the cutoff. No tools, images, web, fighter identities, fight memories, external data, or future events. These are model-generated annotations and may contain errors. Predict the phase AT THE END of the next four seconds and whether each event occurs ANYWHERE within those next four seconds. Give calibrated probabilities, accounting for continuation/persistence and uncertainty; do not turn a plausible story into certainty. Return valid JSON without markdown.\n'+f'Cutoff={t}s; forecast interval=[{t},{t+4})s; condition=reviewed_history.\nAnnotation prefix:\n'+json.dumps(data)+'\n'+SCHEMA
 tag=f'predict_{c}_{t:02d}_reviewed_history_r1';obj=call(tag,prompt);obj.update(clip=c,cutoff=t,horizon=4,condition='reviewed_history',repeat=1,input_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest());(OUT/'predictions'/f'{tag}.json').write_text(json.dumps(obj,indent=2));print('REVIEWED PREDICT',c,t,flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(task,[(c,t) for c in 'ABCD' for t in [4,8,12,16]]))
