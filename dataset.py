"""MUSDB18-HQ WAV loading; never downloads or redistributes music."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from model import RATE, SOURCES

# Official MUSDB validation list, sigsep/sigsep-mus-db musdb/configs/mus.yaml.
VALIDATION = {
    'Actions - One Minute Smile', 'Clara Berry And Wooldog - Waltz For My Victims',
    'Johnny Lokke - Promises & Lies', 'Patrick Talbot - A Reason To Leave',
    'Triviul - Angelsaint', 'Alexander Ross - Goodbye Bolero', 'Fergessen - Nos Palpitants',
    'Leaf - Summerghost', 'Skelpolu - Human Mistakes', 'Young Griffo - Pennies',
    'ANiMAL - Rockshow', 'James May - On The Line', 'Meaxic - Take A Step',
    'Traffic Experiment - Sirens',
}

def discover(root, split):
    root=Path(root)
    subset=root/('test' if split=='test' else 'train')
    if not subset.is_dir():
        raise FileNotFoundError(f'Missing {subset}; see TRAINING.md for dataset setup')
    dirs=sorted(p for p in subset.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if split!='test':
        missing=VALIDATION-{p.name for p in dirs}
        if missing: raise ValueError(f'Missing standard validation tracks: {sorted(missing)}')
        dirs=[p for p in dirs if (p.name in VALIDATION)==(split=='val')]
    tracks=[]
    for d in dirs:
        lengths=[]
        for name in (*SOURCES,'mixture'):
            path=d/f'{name}.wav'
            info=sf.info(str(path))
            if info.samplerate!=RATE or info.channels!=2:
                raise ValueError(f'Expected stereo {RATE} Hz: {path}')
            lengths.append(info.frames)
        if len(set(lengths))!=1 or min(lengths)==0:
            raise ValueError(f'Misaligned or empty files: {d}: {lengths}')
        tracks.append((d,lengths[0]))
    if not tracks: raise ValueError(f'No tracks in {subset} for {split}')
    return tracks

def read_wave(path,start=0,frames=-1):
    x,sr=sf.read(str(path),start=start,frames=frames,dtype='float32',always_2d=True)
    if sr!=RATE or x.shape[1]!=2: raise ValueError(f'Invalid WAV: {path}')
    return np.ascontiguousarray(x.T)

def read_crop(track,source,start,n):
    d,length=track
    x=read_wave(d/f'{source}.wav',start,min(n,length-start))
    return np.pad(x,((0,0),(0,n-x.shape[1])))

class TrainClips(Dataset):
    def __init__(self,tracks,n,count,seed,epoch,offset=0,remix=.5,gain_db=6.):
        self.tracks,self.n,self.count,self.seed,self.epoch,self.offset=tracks,n,count,seed,epoch,offset
        self.remix,self.gain_db=remix,gain_db
    def __len__(self): return self.count-self.offset
    def __getitem__(self,i):
        # Independent per-index RNG makes resume and worker count independent of sample choices.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i+self.offset]))
        base=self.tracks[int(rng.integers(len(self.tracks)))]
        start=int(rng.integers(max(0,base[1]-self.n)+1))
        mix_sources=rng.random()<self.remix
        xs=[]
        for source in SOURCES:
            tr=base if not mix_sources else self.tracks[int(rng.integers(len(self.tracks)))]
            pos=start if not mix_sources else int(rng.integers(max(0,tr[1]-self.n)+1))
            x=read_crop(tr,source,pos,self.n)
            xs.append(x*np.float32(10**(rng.uniform(-self.gain_db,self.gain_db)/20)))
        stems=np.stack(xs).astype('float32')
        # Common scaling preserves the additive mix relationship.
        peak=max(float(np.max(np.abs(stems.sum(0)))),float(np.max(np.abs(stems))),.98)
        stems*=np.float32(.98/peak)
        return torch.from_numpy(stems)

class ValidationClips(Dataset):
    def __init__(self,tracks,n,per_track=2):
        self.n=n
        self.items=[(tr,int(pos)) for tr in tracks for pos in
                    np.linspace(0,max(0,tr[1]-n),per_track,dtype=int)]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        tr,pos=self.items[i]
        mix=read_crop(tr,'mixture',pos,self.n)
        stems=np.stack([read_crop(tr,s,pos,self.n) for s in SOURCES])
        return torch.from_numpy(mix),torch.from_numpy(stems)
