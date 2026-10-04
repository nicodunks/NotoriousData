import os
from pathlib import Path
import json,time,hashlib,subprocess,sys,platform
import cv2,numpy as np
import torch, onnxruntime as ort
from sam2.sam2_image_predictor import SAM2ImagePredictor
from rtmlib import RTMPose

ROOT=Path.cwd(); OUT=ROOT/'outputs/expanded/segmentation'; WORK=ROOT/'work/expanded/segmentation'; OUT.mkdir(parents=True,exist_ok=True)
SAM_ID='facebook/sam2.1-hiera-tiny'
RT=Path(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'))
GUIDE=ROOT/'outputs/expanded/identity'
CLIPS=ROOT/'outputs/experiment/clips'
# Manually selected torso prompts from original 960x540 frames. These are prompts, not labels/ground truth.
PROMPTS={
 ('C',0): {'fighter_black':{'pos':[430,405],'neg':[440,140]},'fighter_red':{'pos':[440,140],'neg':[430,405]}},
 ('C',48):{'fighter_black':{'pos':[505,330],'neg':[410,180]},'fighter_red':{'pos':[410,180],'neg':[505,330]}},
 ('D',0): {'fighter_black':{'pos':[310,270],'neg':[185,340]},'fighter_red':{'pos':[185,340],'neg':[310,270]}},
 ('D',48):{'fighter_black':{'pos':[505,190],'neg':[450,280]},'fighter_red':{'pos':[450,280],'neg':[505,190]}},
}
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def choose(masks,scores,points,labels):
    # choose highest SAM IoU estimate among candidates obeying user prompts; otherwise highest score and flag conflict
    p=np.asarray(points,int); lab=np.asarray(labels)
    ok=[]
    for i,m in enumerate(masks):
        inside=[bool(m[y,x]) for (x,y) in p]
        good=all(v for v,l in zip(inside,lab) if l==1) and not any(v for v,l in zip(inside,lab) if l==0)
        if good: ok.append(i)
    pool=ok or list(range(len(scores)))
    idx=max(pool,key=lambda i:float(scores[i]))
    return idx,bool(ok)
def pose_render(im,people,color,title):
    canvas=im.copy()
    for p in people:
        kp=np.asarray(p['keypoints']); col=p['bgr']
        valid=(kp[:,0]>=0)&(kp[:,0]<im.shape[1])&(kp[:,1]>=0)&(kp[:,1]<im.shape[0])&(kp[:,2]>1.5)
        for a,b in EDGES:
            if valid[a] and valid[b]: cv2.line(canvas,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),col,2,cv2.LINE_AA)
        for j in range(23):
            if valid[j]:cv2.circle(canvas,tuple(kp[j,:2].astype(int)),3,col,-1,cv2.LINE_AA)
        x1,y1,x2,y2=np.asarray(p['box']).astype(int);cv2.rectangle(canvas,(x1,y1),(x2,y2),col,1)
        cv2.putText(canvas,p['identity'],(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,col,1,cv2.LINE_AA)
    cv2.putText(canvas,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return canvas

def row_time(guide, frame):
    return next(float(x['time']) for x in guide['frames'] if x['frame']==frame)

def main():
    start=time.time(); print('load sam',flush=True)
    predictor=SAM2ImagePredictor.from_pretrained(SAM_ID,device='mps' if torch.backends.mps.is_available() else 'cpu')
    samload=time.time()-start
    model=RTMPose(str(RT),model_input_size=(288,384),backend='onnxruntime',device='cpu')
    opts=ort.SessionOptions(); opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
    model.session=ort.InferenceSession(str(RT),sess_options=opts,providers=['CPUExecutionProvider'])
    guide_data={s:json.loads((GUIDE/f'{s}-guided-poses.json').read_text()) for s in 'CD'}
    frames=[]; contact=[]
    for sid,fr in [('C',0),('C',48),('D',0),('D',48)]:
        t0=time.time(); clip=CLIPS/f'{sid}.mp4'; cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS); sample_time=float(row_time(guide_data[sid],fr));cap.set(cv2.CAP_PROP_POS_MSEC,sample_time*1000);ok,bgr=cap.read();cap.release()
        if not ok: raise RuntimeError(f'cannot read {clip} frame {fr}')
        h,w=bgr.shape[:2]; assert (w,h)==(960,540),(w,h)
        row=next(x for x in guide_data[sid]['frames'] if x['frame']==fr)
        fighters=[x for x in row['people'] if x['identity'] in ('fighter_black','fighter_red')]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); predictor.set_image(rgb)
        pred={}; masks={}; t_sam=time.time()
        for fighter in fighters:
            key=fighter['identity']; box=np.asarray(fighter['box'],np.float32)
            prompts=PROMPTS[(sid,fr)][key]; points=np.asarray([prompts['pos'],prompts['neg']],np.float32); labels=np.array([1,0],np.int32)
            masks3,scores,logits=predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=True)
            ix,consistent=choose(masks3,scores,points,labels); mask=np.asarray(masks3[ix],bool); masks[key]=mask
            maskfile=OUT/f'{sid}-{fr:03d}-{key}-sam-mask.png'; cv2.imwrite(str(maskfile),(mask*255).astype('uint8'))
            pred[key]={'identity':key,'box':box.tolist(),'points':{'positive_torso':prompts['pos'],'negative_opponent_torso':prompts['neg']},'candidate_scores':np.asarray(scores).astype(float).tolist(),'selected_candidate':int(ix),'prompt_consistent_candidate_available':consistent,'mask_area_px':int(mask.sum()),'mask_area_fraction':float(mask.mean()),'mask_path':str(maskfile.relative_to(ROOT)),'candidate_masks':[]}
            for j,m in enumerate(masks3):
                ppos=prompts['pos'];pneg=prompts['neg'];pred[key]['candidate_masks'].append({'index':j,'area_px':int(np.asarray(m,bool).sum()),'positive_point_inside':bool(m[ppos[1],ppos[0]]),'negative_point_inside':bool(m[pneg[1],pneg[0]])})
        samtime=time.time()-t_sam
        overlap=masks['fighter_black']&masks['fighter_red']; overlap_px=int(overlap.sum())
        overlay=bgr.copy(); overlay[masks['fighter_black']]=(.65*overlay[masks['fighter_black']]+.35*np.array([255,190,20])).astype(np.uint8);overlay[masks['fighter_red']]=(.65*overlay[masks['fighter_red']]+.35*np.array([20,80,255])).astype(np.uint8)
        # Retain original pixels except opponent-owned mask; zeroing opponent can erase target portions at contact/overlap.
        baseline_people=[]; separated_people=[]; baseline_time=0; separated_time=0
        for fighter in fighters:
            key=fighter['identity']; col=(0,220,255) if key=='fighter_black' else (255,100,140); box=np.asarray(fighter['box'],np.float32)
            a=time.time(); xy,sc=model(bgr,[box]); baseline_time+=time.time()-a
            basekp=np.column_stack((xy[0],sc[0]))
            baseline_people.append({'identity':key,'box':box.tolist(),'keypoints':basekp,'bgr':col})
            opponent='fighter_red' if key=='fighter_black' else 'fighter_black'
            masked=bgr.copy(); masked[masks[opponent]]=np.median(bgr.reshape(-1,3),axis=0).astype(np.uint8)
            a=time.time(); xy2,sc2=model(masked,[box]); separated_time+=time.time()-a
            kp=np.column_stack((xy2[0],sc2[0])); separated_people.append({'identity':key,'box':box.tolist(),'keypoints':kp,'bgr':col})
            pred[key]['baseline_keypoints']=basekp.round(3).tolist();pred[key]['opponent_masked_keypoints']=kp.round(3).tolist();pred[key]['pose_delta_px_median_23']=float(np.median(np.linalg.norm(basekp[:23,:2]-kp[:23,:2],axis=1)))
        panels=[bgr.copy(),overlay,pose_render(bgr,baseline_people,(0,0,0),f'{sid}{fr}: baseline RTMW candidates'),pose_render(bgr,separated_people,(0,0,0),f'{sid}{fr}: opponent mask removed')]
        labels=['Original + inherited boxes','SAM masks: cyan black / red pink','Baseline RTMW (candidate ownership)','RTMW after removing opponent mask']
        for im,label in zip(panels,labels):cv2.putText(im,label,(7,520),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        panel_path=OUT/f'{sid}-{fr:03d}-comparison.jpg';cv2.imwrite(str(panel_path),np.hstack(panels))
        # standalone candidate overlays and mask overlay saved without altering originals
        cv2.imwrite(str(OUT/f'{sid}-{fr:03d}-sam-overlay.jpg'),overlay)
        frames.append({'clip':sid,'frame_index':fr,'time_seconds':sample_time,'sample_index_12hz':fr,'source_native_frame_approx':int(round(sample_time*fps)),'input':str(clip.relative_to(ROOT)),'width':w,'height':h,'fps':fps,'fighters':pred,'mask_overlap_px':overlap_px,'mask_overlap_fraction_of_union':float(overlap.sum()/max(1,(masks['fighter_black']|masks['fighter_red']).sum())),'timing_seconds':{'sam_predict_both':samtime,'baseline_rtmw_both':baseline_time,'masked_rtmw_both':separated_time,'total_frame':time.time()-t0},'comparison_contact_sheet':str(panel_path.relative_to(ROOT))})
        print(f'{sid} {fr}: masks {[(k,v["mask_area_px"]) for k,v in pred.items()]}, overlap={overlap_px}, timing={frames[-1]["timing_seconds"]}',flush=True)
    # compact contact sheet grid, downscaled panels
    sheets=[]
    for item in frames:
        im=cv2.imread(str(ROOT/item['comparison_contact_sheet']));im=cv2.resize(im,(1440,405));cv2.putText(im,f"{item['clip']} {item['time_seconds']:.3f}s",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2);sheets.append(im)
    contact=np.vstack(sheets); cv2.imwrite(str(OUT/'sam2-rtmw-contact-sheet.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,92])
    # checkpoint provenance
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
    conf,ckpt=HF_MODEL_ID_TO_FILENAMES[SAM_ID]
    ckpath=Path(hf_hub_download(repo_id=SAM_ID,filename=ckpt))
    try:
        git=subprocess.check_output(['git','-C',str(Path(__import__('sam2').__file__).parent),'rev-parse','HEAD'],text=True).strip()
    except Exception:git='unknown'
    out={'experiment':'SAM2.1-tiny prompted segmentation to isolate opponent pixels, followed by pretrained RTMW-133 candidate pose inference','no_training':True,'limitations':['SAM masks are prompted segmentation candidates, not fighter identity ground truth.','Positive/negative prompt coordinates and boxes were manually selected from frames and inherited from separate Sol-guided box annotations.','Opponent-mask removal may erase occluded parts of the target; masked pose output is a diagnostic candidate, not a claim of improved ownership.','SAM quality scores are estimates, not calibrated probabilities.','Sparse samples at 0 and 4 seconds only; no temporal tracking or identity propagation.'],'software':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'mps_available':torch.backends.mps.is_available(),'sam2_repo_commit':git,'sam_model_id':SAM_ID,'sam_checkpoint':str(ckpath),'sam_checkpoint_sha256':sha(ckpath),'sam_checkpoint_bytes':ckpath.stat().st_size,'sam_load_seconds':samload,'rtmw_checkpoint':str(RT),'rtmw_checkpoint_sha256':sha(RT),'rtmw_provider':'CPUExecutionProvider','rtmw_threads':{'intra_op':6,'inter_op':1},'docs':['https://github.com/facebookresearch/sam2','https://huggingface.co/facebook/sam2.1-hiera-tiny']},'frames':frames}
    dest=OUT/'sam2-rtmw-c-d-0-4.json'; dest.write_text(json.dumps(out,indent=2));print('WROTE',dest,flush=True)
if __name__=='__main__':main()
