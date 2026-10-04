import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json,numpy as np
O=Path('outputs/expanded');s=json.loads((O/'key_moments/subtypes/scores.json').read_text());rows=[]
for phase,target,label in [('standing','punch_attempt','Standing punch'),('standing','kick_attempt','Standing kick'),('clinch','punch_attempt','Clinch punch'),('clinch','knee_elbow_attempt','Clinch knee / elbow'),('ground','ground_strike_attempt','Ground strike · frozen')]:
 r=next(r for r in s if r['subset']=='ordinary_registered' and r['condition']=='rich_history_textonly' and r['past_phase']==phase and r['target']==target);rows.append((label,r['positives'],r['negatives'],r['unclear'],r['true_alerts']))
rows.append(('Ground strike · native review',3,0,5,1))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11});fig,ax=plt.subplots(figsize=(12,6.5));fig.patch.set_facecolor('#10171c');ax.set_facecolor('#10171c');ys=np.arange(len(rows));left=np.zeros(len(rows))
for index,color,label in [(1,'#22d3ee','Positive reference'),(2,'#a3c29b','Negative reference'),(3,'#667581','Unclear reference')]:
 vals=np.array([r[index] for r in rows]);ax.barh(ys,vals,left=left,color=color,height=.62,label=label)
 for y,v,l in zip(ys,vals,left):
  if v:ax.text(l+v/2,y,str(v),ha='center',va='center',color='#10171c' if index!=3 else 'white',fontweight='bold')
 left+=vals
for y,r in zip(ys,rows):ax.text(8.4,y,f'{r[4]}/{r[1]} positives alerted',va='center',color='#edf0e9')
ax.set_yticks(ys,[r[0] for r in rows],color='#edf0e9');ax.invert_yaxis();ax.set_xlim(0,11.8);ax.set_xticks(range(0,9),labels=range(0,9),color='#a3b0b7');ax.set_xlabel('Selected forecast windows — overlapping within bouts',color='#a3b0b7');ax.tick_params(axis='y',length=0)
for sp in ax.spines.values():sp.set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.47,-.16),ncol=3,frameon=False,labelcolor='#edf0e9');fig.text(.04,.95,'What the prediction evidence actually contains',fontsize=21,color='#edf0e9');fig.text(.04,.895,'Past annotations only → next four seconds. Alert threshold 50%. No training.',color='#a3b0b7');fig.text(.04,.035,'Tiny selected samples; same-model references, not expert truth. Ground false alarms untested without negatives.\nNative review is a targeted sensitivity analysis. Standing punch/kick each also had one false alert.',color='#a3b0b7',fontsize=10)
fig.subplots_adjust(left=.28,right=.98,top=.82,bottom=.23);fig.savefig(O/'assessment/prediction-evidence.png',dpi=160,facecolor=fig.get_facecolor());fig.savefig(O/'assessment/prediction-evidence.pdf',facecolor=fig.get_facecolor());print('PREDICTION EVIDENCE FIGURE SAVED')
