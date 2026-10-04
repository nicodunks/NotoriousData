import sys,time,subprocess
from pathlib import Path
R=Path.cwd();O=R/'outputs/experiment';B=R/'work/experiment'
while not all((O/'predictions'/f'predict_{c}_{t:02d}_reviewed_history_r1.json').exists() for c in 'ABCD' for t in [4,8,12,16]):time.sleep(5)
subprocess.run([sys.executable,str(B/'pose_predict.py')],check=True)
subprocess.run([sys.executable,str(B/'validate.py')],check=True)
subprocess.run([sys.executable,str(B/'score.py')],check=True)
subprocess.run([sys.executable,str(B/'paired.py')],check=True)
print('WHOLE-BODY EXPERIMENT COMPLETE',flush=True)
