"""Two additional ordinary 20s sequences from two newly downloaded official broadcasts."""
import concurrent.futures,json,hashlib,sys
from pathlib import Path
from annotate import prepare,annotate,S,O
from action_crops import run as crop_review

def prepare_extra():
 prepare([s for s in S if s[0] in 'IJ'],'annotation_manifest_extra.json')
 p=O/'annotation_manifest_extra.json';a=json.loads(p.read_text());scouts={'I':'S39','J':'S50'}
 for m in a:
  scout=scouts[m['id']];elig=O/'scouting'/f'{scout}.json'
  m['selection']='Ordinary mid-round 20-second interval chosen from systematic scouting; confirmed active footage with no replay flag. Not chosen for known finish or future outcome.'
  m['selection_scout_id']=scout;m['eligibility_annotation']=str(elig);m['eligibility_annotation_sha256']=hashlib.sha256(elig.read_bytes()).hexdigest()
  m['clip_sha256']=hashlib.sha256(Path(m['clip']).read_bytes()).hexdigest()
  m['limits']=['No source title, date, outcome or scouting labels passed to event annotators. Visual scoreboard can identify footage; incomplete visual blinding.','2Hz discovery plus selected native-rate windows; not every contact is verified.','No training, independent-human truth or gold accuracy claim.','Requested frame timestamps are quantized to native decoded frames (~33ms).','Native evidence files preserve pixels but model frontend may rescale large sheets.']
 p.write_text(json.dumps(a,indent=2))
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare_extra()
 else:
  assert all((O/'annotations'/f'{c}_detail.json').exists() for c in 'EFGH'),'Finish E-H fine detail before adding sources'
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(annotate,'IJ'))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(crop_review,'IJ'))
