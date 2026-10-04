import time,sys,subprocess,json
from pathlib import Path
ROOT=Path.cwd();O=ROOT/'outputs/experiment';B=ROOT/'work/experiment'
while not all((O/'verification'/f'{c}_{t:02d}.json').exists() for c in 'ABCD' for t in range(0,20,4)):time.sleep(5)
subprocess.run([sys.executable,str(B/'adjudicate.py')],check=True)
subprocess.run([sys.executable,str(B/'reviewed_predict.py')],check=True)
subprocess.run([sys.executable,str(B/'validate.py')],check=True)
subprocess.run([sys.executable,str(B/'score.py')],check=True)
print('ALL EXPERIMENT CALLS COMPLETE AND SCORED',flush=True)
