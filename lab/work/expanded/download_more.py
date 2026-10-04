import json,subprocess,cv2,hashlib,time
from pathlib import Path
R=Path.cwd();O=R/'outputs/expanded';catalog=json.loads((O/'source_catalog.json').read_text());results=[]
for s in catalog['sources']:
 p=R/'work/raw'/f"{s['video_id']}.mp4"; log=R/'work/expanded'/f"download-{s['video_id']}.log";cmd=[str(R/'work/venv/bin/yt-dlp'),'--no-playlist','--newline','--progress-delta','10','--write-info-json','--js-runtimes','node:node','-f','bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]','-o',str(p.with_suffix('.%(ext)s')),s['source_url']]
 if not p.exists():
  with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 else:r=None
 row={**s,'download_attempted':True,'return_code':None if r is None else r.returncode}
 if p.exists():
  c=cv2.VideoCapture(str(p));ok,f=c.read();row.update(download_path=str(p),decoded_first_frame=ok,width=int(c.get(3)),height=int(c.get(4)),fps=c.get(5),decoded_duration=c.get(7)/c.get(5) if c.get(5) else None,sha256=hashlib.sha256(p.read_bytes()).hexdigest());c.release()
 else:row['error_tail']=log.read_text()[-1600:]
 results.append(row);(O/'download_results.json').write_text(json.dumps(results,indent=2));print('DOWNLOAD',s['video_id'],row.get('decoded_first_frame'),row.get('decoded_duration'),flush=True)
print('DOWNLOADS COMPLETE',flush=True)
