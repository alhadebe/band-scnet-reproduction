"""Resumable MUSDB18-HQ training. See TRAINING.md."""
import argparse, math, os, signal, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import BandSCNet, encode, decode, RATE, SOURCES
from dataset import discover, TrainClips, ValidationClips

def atomic_save(state,path):
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp); os.replace(tmp,path)

def loss_fn(pred,target):
    # RMSE of real/imaginary components, each source equally represented.
    return (pred.float().sub(target.float()).square().mean()+1e-8).sqrt()

@torch.no_grad()
def validate(model,loader,device):
    model.eval(); losses=[]; scores=[]
    for mix,stems in loader:
        mix,stems=mix.to(device),stems.to(device)
        b,_,_,n=stems.shape
        pred=model(encode(mix))
        losses.append(loss_fn(pred,encode(stems.reshape(b,8,n))).item())
        y=decode(pred.float(),n)
        # Clip-level global SDR proxy, not the paper's museval metric.
        ref=stems.square().sum(dim=(-2,-1))
        err=(stems-y).square().sum(dim=(-2,-1))
        s=10*torch.log10((ref+1e-8)/(err+1e-8))
        s=s.masked_fill(ref<1e-6,float('nan'))
        scores.append(s.cpu().numpy())
    return float(np.mean(losses)),np.nanmean(np.concatenate(scores),axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/dataset'))
    p.add_argument('--run',type=Path,default=Path('/checkpoints/bandscnet'))
    p.add_argument('--device',choices=['cpu','cuda'],default=os.getenv('DEVICE','cuda'))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--steps',type=int,default=500,help='Optimizer updates per epoch')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--accum',type=int,default=4)
    p.add_argument('--seconds',type=float,default=11)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--remix',type=float,default=.5)
    p.add_argument('--gain-db',type=float,default=6.)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--val-clips',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check-data',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.steps,a.batch,a.accum,a.seconds,a.layers,a.lr,a.threads,a.save_every,a.val_clips)<=0 or a.workers<0 or not 0<=a.remix<=1 or a.gain_db<0:
        p.error('Invalid training configuration')
    tr=discover(a.data,'train'); va=discover(a.data,'val')
    print(f'Dataset: {len(tr)} training tracks, {len(va)} validation tracks')
    if a.check_data:
        te=discover(a.data,'test')
        print(f'Test tracks: {len(te)}; dataset metadata check complete')
        if (len(tr),len(va),len(te))!=(86,14,50): raise ValueError('Expected full MUSDB18-HQ split 86/14/50')
        return
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if a.precision!='fp32' and device.type!='cuda': raise ValueError('AMP requires CUDA for this trainer')
    if a.precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('BF16 not supported; use fp32')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    if device.type=='cuda': torch.backends.cudnn.benchmark=False
    n=round(a.seconds*RATE)
    model=BandSCNet(a.layers).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=.5,patience=5)
    scaler=torch.amp.GradScaler('cuda',enabled=a.precision=='fp16')
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float16
    cfg={k:getattr(a,k) for k in ('steps','batch','accum','seconds','layers','lr','precision','seed','remix','gain_db','val_clips')}
    cfg['tracks']=[d.name for d,_ in tr]
    a.run.mkdir(parents=True,exist_ok=True)
    latest=a.run/'latest.pt'; bestpath=a.run/'best.pt'
    epoch=0; completed=0; global_step=0; best=float('inf')
    if a.resume:
        if not latest.exists(): raise FileNotFoundError(latest)
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if cfg!=state['config']: raise ValueError('Resume configuration mismatch; use the original flags or a new --run directory')
        model.load_state_dict(state['model']); optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        epoch,completed,global_step,best=(state[k] for k in ('epoch','completed','global_step','best'))
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and 'cuda_rng' in state: torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resuming epoch {epoch+1}, {completed}/{a.steps} updates, global step {global_step}')
    elif latest.exists() or bestpath.exists():
        raise FileExistsError(f'{a.run} already has checkpoints; use --resume or another --run')
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    def checkpoint(path):
        state={'format':1,'model':model.state_dict(),'optimizer':optim.state_dict(),
               'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
               'completed':completed,'global_step':global_step,'best':best,'config':cfg,
               'torch_rng':torch.get_rng_state()}
        if device.type=='cuda': state['cuda_rng']=torch.cuda.get_rng_state_all()
        atomic_save(state,path)
    writer=SummaryWriter(str(a.run/'tensorboard'),purge_step=global_step if a.resume else None)
    pin=device.type=='cuda'
    val=DataLoader(ValidationClips(va,n,a.val_clips),batch_size=1,num_workers=a.workers,pin_memory=pin)
    try:
        while epoch<a.epochs and not stop:
            model.train()
            ds=TrainClips(tr,n,a.steps*a.accum*a.batch,a.seed,epoch,
                          offset=completed*a.accum*a.batch,remix=a.remix,gain_db=a.gain_db)
            loader=iter(DataLoader(ds,batch_size=a.batch,num_workers=a.workers,pin_memory=pin))
            while completed<a.steps and not stop:
                start=time.monotonic(); optim.zero_grad(set_to_none=True); total=0.
                for _ in range(a.accum):
                    stems=next(loader).to(device,non_blocking=pin)
                    b,_,_,length=stems.shape
                    target=encode(stems.reshape(b,8,length))
                    mix=encode(stems.sum(1))
                    with torch.autocast(device_type=device.type,dtype=dtype,enabled=a.precision!='fp32'):
                        pred=model(mix)
                    loss=loss_fn(pred,target)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; try fp32 or lower learning rate')
                    scaler.scale(loss/a.accum).backward()
                    total+=loss.item()/a.accum
                scaler.unscale_(optim)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=a.precision!='fp16')
                scaler.step(optim); scaler.update()
                completed+=1; global_step+=1
                writer.add_scalar('train/rmse',total,global_step)
                writer.add_scalar('train/lr',optim.param_groups[0]['lr'],global_step)
                writer.add_scalar('train/grad_norm',float(norm),global_step)
                writer.add_scalar('train/update_seconds',time.monotonic()-start,global_step)
                if global_step==1 or global_step%10==0:
                    print(f'epoch {epoch+1}/{a.epochs} update {completed}/{a.steps} loss {total:.6f} elapsed {time.monotonic()-start:.2f}s')
                if global_step%a.save_every==0 or stop: checkpoint(latest); writer.flush()
            if stop: break
            loss,sdr=validate(model,val,device)
            if not math.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
            scheduler.step(loss)
            writer.add_scalar('validation/rmse',loss,global_step)
            for source,value in zip(SOURCES,sdr): writer.add_scalar('validation/clip_sdr_'+source,float(value),global_step)
            print(f'epoch {epoch+1}: val RMSE {loss:.6f}, clip SDR proxy {dict(zip(SOURCES,map(float,sdr)))}')
            epoch+=1; completed=0
            if loss<best:
                best=loss; checkpoint(bestpath)
            checkpoint(latest); writer.flush()
        if stop: checkpoint(latest); print('Stop requested; saved completed optimizer boundary')
    finally: writer.close()

if __name__=='__main__': main()
