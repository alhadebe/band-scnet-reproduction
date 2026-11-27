"""Audit, benchmark and export an independent untrained Band-SCNet reproduction."""
import argparse, json, os, statistics, time
from pathlib import Path
import numpy as np
import torch
from model import BandSCNet, encode, decode, RATE, BINS, NFFT, HOP, SOURCES


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def timed(fn, device, warmup, repeats):
    for _ in range(warmup): fn(); sync(device)
    values=[]
    for _ in range(repeats):
        sync(device); start=time.perf_counter(); fn(); sync(device)
        values.append(time.perf_counter()-start)
    return {'seconds':values,'median_seconds':statistics.median(values)}


def save(data, directory, name):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


def ort_session(path, device, threads):
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers=['CPUExecutionProvider']
    if device.type=='cuda':
        # Import torch first and preload its CUDA/cuDNN libraries where supported.
        if hasattr(ort,'preload_dlls'): ort.preload_dlls()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime CUDA provider unavailable')
        providers.insert(0,'CUDAExecutionProvider')
    sess=ort.InferenceSession(str(path),sess_options=options,providers=providers)
    if device.type=='cuda' and sess.get_providers()[0]!='CUDAExecutionProvider':
        raise RuntimeError(f'CUDA session did not initialize: {sess.get_providers()}')
    return sess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','bench','export','onnx','train-smoke'])
    p.add_argument('--seconds',type=float,default=11)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--device',default=os.getenv('DEVICE','cpu'),choices=['cpu','cuda'])
    p.add_argument('--output-dir',type=Path,default=Path(os.getenv('OUTPUT_DIR','output')))
    p.add_argument('--model-dir',type=Path,default=Path(os.getenv('MODEL_DIR','models')))
    p.add_argument('--profile-flops',action='store_true')
    p.add_argument('--checkpoint',type=Path,help='Load trusted local training checkpoint; use a separate --model-dir for ONNX')
    args=p.parse_args()
    if args.seconds<=0 or args.threads<1 or args.layers<1 or args.repeats<1 or args.warmup<0: p.error('positive duration, threads, layers and repeats required; warmup >= 0')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); torch.manual_seed(42)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable in PyTorch')
    # Avoid TF32 differences in the reference/parity tests.
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model=BandSCNet(args.layers).to(device).eval()
    if args.checkpoint:
        if args.action not in ('export','onnx','bench'): p.error('--checkpoint is for export, onnx or bench')
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if state['config']['layers']!=args.layers: p.error('Checkpoint layer count differs; set --layers')
        model.load_state_dict(state['model'])
    n=max(1,round(args.seconds*RATE)); duration=n/RATE
    wave=torch.randn(1,2,n,device=device)*.1
    spec=encode(wave)
    params=sum(p.numel() for p in model.parameters())
    result={'action':args.action,'untrained':not bool(args.checkpoint),'checkpoint':str(args.checkpoint) if args.checkpoint else None,'device':str(device),'torch':torch.__version__,
            'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else os.uname().machine,
            'threads':args.threads,'layers':args.layers,'parameters':params,'sample_rate':RATE,
            'input_seconds':duration,'frequency_bins':model.frequencies,'input_shape':list(spec.shape),
            'source_order':SOURCES,'precision':'fp32'}
    if args.action=='audit':
        with torch.inference_mode():
            pred=model(spec); stems=decode(pred,n)
            assert pred.shape==(1,16,spec.shape[2],BINS)
            assert stems.shape==(1,4,2,n) and torch.isfinite(stems).all()
            check=torch.randn(1,4,24,BINS,device=device)
            altered=check.clone(); altered[:,:,12:]=torch.randn_like(altered[:,:,12:])
            diff=(model(check)[:,:,:12]-model(altered)[:,:,:12]).abs().max().item()
            err=(decode(spec,n)[:,0]-wave).abs().max().item()
            assert diff<1e-5 and err<1e-5,(diff,err)
        result.update(output_shape=list(pred.shape),wave_shape=list(stems.shape),finite=True,
                      causal_prefix_max_abs_error=diff,stft_roundtrip_max_abs_error=err)
    elif args.action=='bench':
        with torch.inference_mode():
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            network=timed(lambda:model(spec),device,args.warmup,args.repeats)
            pipeline=timed(lambda:decode(model(encode(wave)),n),device,args.warmup,args.repeats)
            result.update(network=network,pipeline=pipeline,rtf=pipeline['median_seconds']/duration,
                          speed_x=duration/pipeline['median_seconds'])
            if device.type=='cuda': result['peak_torch_allocated_mib']=torch.cuda.max_memory_allocated(device)/2**20
            if args.profile_flops:
                from torch.utils.flop_counter import FlopCounterMode
                with FlopCounterMode(display=False) as fc: model(spec)
                result['counted_gflops_per_audio_second']=fc.get_total_flops()/duration/1e9
    elif args.action=='train-smoke':
        # Synthetic four-target examples exercise autograd only; not a real training run.
        model.train(); optim=torch.optim.Adam(model.parameters(),lr=5e-4)
        targets=torch.randn(1,4,2,n,device=device)*.03
        mix=targets.sum(1)
        target_spec=encode(targets.reshape(1,8,n))
        pred=model(encode(mix)); loss=((pred-target_spec).square().mean()+1e-8).sqrt()
        optim.zero_grad(); loss.backward()
        finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        assert finite,'missing or nonfinite gradients'
        optim.step(); result.update(loss=float(loss.detach()),all_gradients_finite=finite)
    else:
        # Static input shape makes the first ONNX graph straightforward to validate.
        tag='trained-' if args.checkpoint else ''
        filename=f'bandscnet-{tag}l{args.layers}-t{spec.shape[2]}.onnx'; path=args.model_dir/filename
        if args.action=='export':
            import onnx
            args.model_dir.mkdir(parents=True,exist_ok=True)
            # Always export reference in CPU FP32, independent of requested device.
            ref=model.cpu().eval(); inp=spec.cpu()
            with torch.no_grad():
                torch.onnx.export(ref,(inp,),str(path),input_names=['mix_spec'],output_names=['stems_spec'],opset_version=17,dynamo=False)
                expected=ref(inp).numpy()
            onnx.checker.check_model(str(path))
            session=ort_session(path,device,args.threads)
            actual=session.run(['stems_spec'],{'mix_spec':inp.numpy()})[0]
            max_abs=float(np.max(np.abs(actual-expected)))
            rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
            assert actual.shape==expected.shape and np.isfinite(actual).all() and rel<1e-3, (max_abs,rel)
            result.update(onnx=str(path),onnx_bytes=path.stat().st_size,providers=session.get_providers(),max_abs_error=max_abs,relative_l2_error=rel)
        else:
            if not path.exists(): raise FileNotFoundError(f'Export first using identical --seconds and --layers: {path}')
            session=ort_session(path,device,args.threads)
            # Host-side FFT + host I/O includes CPU/GPU transfer costs. ORT handles inference.
            w=wave.cpu(); a=spec.cpu().numpy()
            def net(): return session.run(['stems_spec'],{'mix_spec':a})[0]
            def pipeline():
                pred=session.run(['stems_spec'],{'mix_spec':encode(w).numpy()})[0]
                return decode(torch.from_numpy(pred),n)
            network=timed(net,torch.device('cpu'),args.warmup,args.repeats)
            full=timed(pipeline,torch.device('cpu'),args.warmup,args.repeats)
            result.update(onnx=str(path),providers=session.get_providers(),network=network,pipeline=full,rtf=full['median_seconds']/duration,speed_x=duration/full['median_seconds'])
    tag='trained-' if args.checkpoint else ''
    save(result,args.output_dir,f'{tag}{args.action}-{args.device}-l{args.layers}-t{spec.shape[2]}-threads{args.threads}.json')

if __name__=='__main__': main()
