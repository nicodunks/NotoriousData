from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import re,shutil
ROOT=Path(__file__).resolve().parent.parent/'outputs'
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**k):super().__init__(*a,directory=str(ROOT),**k)
 def log_message(self,*a):pass
 def send_head(self):
  path=Path(self.translate_path(self.path));rng=self.headers.get('Range');self.remaining=None
  if not rng or not path.is_file():return super().send_head()
  size=path.stat().st_size;m=re.fullmatch(r'bytes=(\d*)-(\d*)',rng)
  if not m:return super().send_head()
  a,b=m.groups();start=int(a) if a else max(0,size-int(b));end=min(size-1,int(b)) if a and b else size-1
  if start>=size or end<start:self.send_error(416);return None
  f=path.open('rb');f.seek(start);self.remaining=end-start+1;self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Content-Length',str(self.remaining));self.end_headers();return f
 def end_headers(self):
  self.send_header('Accept-Ranges','bytes');super().end_headers()
 def copyfile(self,source,outputfile):
  if self.remaining is None:return shutil.copyfileobj(source,outputfile)
  left=self.remaining
  while left:
   chunk=source.read(min(left,65536))
   if not chunk:break
   try:outputfile.write(chunk)
   except (BrokenPipeError,ConnectionResetError):break
   left-=len(chunk)
ThreadingHTTPServer(('127.0.0.1',8767),Handler).serve_forever()
