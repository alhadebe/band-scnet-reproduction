"""Full-track research evaluation and listening samples using a trained checkpoint."""
import argparse, json, os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from model import BandSCNet, encode, decode, RATE, SOURCES
from dataset import discover, read_wave

@torch.inference_mode()
def separate(model,wave,device,seconds=11,overlap=2):
    n=wave.shape[-1]; size=round(seconds*RATE); fade=round(overlap*RATE)
    if size<1 or not 0<=fade<size: raise ValueError('Require 0 <= overlap < seconds')
    step=size-fade
    out=np.zeros((4,2,n),np.float32); weights=np.zeros(n,np.float32)
    for start in range(0,n,step):
        end=min(n,start+size); length=end-start
        x=np.pad(wave[:,start:end],((0,0),(0,size-length)))
        pred=decode(model(encode(torch.from_numpy(x)[None].to(device))),size)[0,:,:,:length].cpu().numpy()
        if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite model output')
        w=np.ones(length,np.float32)
        k=min(fade,length)
        if k and start>0: w[:k]*=np.linspace(0,1,k,dtype=np.float32)
        if k and end<n: w[-k:]*=np.linspace(1,0,k,dtype=np.float32)
        out[:,:,start:end]+=pred*w
        weights[start:end]+=w
        if end==n: break
    if np.any(weights<=0): raise RuntimeError('Overlap-add did not cover track; choose smaller overlap')
    return out/weights

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['evaluate','listen'])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/dataset'))
    p.add_argument('--device',choices=['cpu','cuda'],default=os.getenv('DEVICE','cuda'))
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--track',help='Exact MUSDB track name, or omit for first track in listening mode')
    p.add_argument('--input',type=Path,help='Optional personal 44.1kHz stereo WAV for listening')
    p.add_argument('--seconds',type=float,default=11)
    p.add_argument('--overlap',type=float,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit',type=int,default=0,help='Debug limit on number of tracks; 0 = all')
    p.add_argument('--output',type=Path,default=Path('/output/evaluation'))
    a=p.parse_args()
    if a.limit<0 or a.threads<1: p.error('Invalid limit or threads')
    if a.action=='evaluate' and a.input: p.error('--input is listening only')
    torch.set_num_threads(a.threads)
    device=torch.device(a.device)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=BandSCNet(state['config']['layers']); model.load_state_dict(state['model']); model.to(device).eval()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.input:
        tracks=[(a.input.stem,a.input,None)]
    else:
        ts=discover(a.data,a.split)
        if a.track: ts=[t for t in ts if t[0].name==a.track]
        if not ts: raise ValueError('No matching tracks')
        tracks=[(d.name,d/'mixture.wav',d) for d,_ in ts]
    if a.action=='listen': tracks=tracks[:1]
    elif a.limit: tracks=tracks[:a.limit]
    rows=[]
    for i,(name,path,d) in enumerate(tracks,1):
        print(f'{i}/{len(tracks)} {name}')
        wave=read_wave(path)
        pred=separate(model,wave,device,a.seconds,a.overlap)
        if a.action=='listen':
            dest=a.output/Path(name).name; dest.mkdir(exist_ok=True)
            for source,y in zip(SOURCES,pred):
                sf.write(str(dest/f'{source}.wav'),y.T,RATE,subtype='FLOAT')
            print(f'Wrote four float WAV stems: {dest}')
        else:
            import museval
            ref=np.stack([read_wave(d/f'{s}.wav').T for s in SOURCES])
            # BSS Eval v4, 1-second windows and hops; no permutation of source labels.
            sdr,_,_,_=museval.evaluate(ref,pred.transpose(0,2,1),win=RATE,hop=RATE,mode='v4')
            med=np.nanmedian(np.where(np.isfinite(sdr),sdr,np.nan),axis=1)
            rows.append({'track':name,'sdr':{s:float(v) if np.isfinite(v) else None for s,v in zip(SOURCES,med)}})
    if rows:
        # Report both medians and means across track medians, and the mean of stem medians.
        medians={}; means={}
        for s in SOURCES:
            vals=[r['sdr'][s] for r in rows if r['sdr'][s] is not None]
            medians[s]=float(np.median(vals)) if vals else None
            means[s]=float(np.mean(vals)) if vals else None
        vals=[v for v in medians.values() if v is not None]
        result={'split':a.split,'tracks':len(rows),'checkpoint':str(a.checkpoint),'seconds':a.seconds,
                'overlap':a.overlap,'median_track_sdr_db':medians,'mean_track_sdr_db':means,
                'mean_of_stem_medians_db':float(np.mean(vals)) if vals else None,'per_track':rows}
        path=a.output/f'{a.split}-sdr.json'
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='per_track'},indent=2))

if __name__=='__main__': main()
