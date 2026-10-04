from pathlib import Path
import zipfile,hashlib,json,datetime
R=Path.cwd();O=R/'outputs';Z=O/'mma-evidence-laboratory.zip';files=sorted(p for p in O.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in {'.DS_Store','package_manifest.json'});manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'uncompressed_bytes':sum(p.stat().st_size for p in files),'full_source_movies_included':False,'archive':'mma-evidence-laboratory.zip','open':'expanded/index.html via a local HTTP server; existing server http://127.0.0.1:8767/expanded/index.html'}
(O/'expanded/package_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(O/'expanded/package_manifest.json')
with zipfile.ZipFile(Z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
 for p in files:z.write(p,p.relative_to(O),compress_type=zipfile.ZIP_STORED if p.suffix.lower() in {'.mp4','.jpg','.jpeg','.png','.webp'} else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(Z) as z:
 assert len(z.namelist())==len(files);assert z.testzip() is None
print('ARCHIVE VERIFIED',Z.stat().st_size,'bytes',len(files),'files',flush=True)
