import time,subprocess,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from predict import case, OUT, BASE
while not all((OUT/'labels'/f'{c}.json').exists() and len(json.loads((OUT/'labels'/f'{c}.json').read_text()))==5 for c in 'ABC'):
 time.sleep(5)
tasks=[(c,t,k,1) for c in 'ABCD' for t in [4,8,12,16] for k in ['coarse_history','rich_recent','rich_history']]+[(c,t,'rich_history',2) for c in 'ABCD' for t in [4,8,12,16]]
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(case,tasks))
print('64 PREDICTIONS FROZEN',flush=True)
subprocess.run([sys.executable,str(BASE/'verify.py')],check=True)
print('PIPELINE COMPLETE',flush=True)
