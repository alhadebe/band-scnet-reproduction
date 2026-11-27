"""Independent Band-SCNet-inspired implementation, based on Yang et al. 2025.
Tensor convention: [batch, real-valued features, time, frequency].
See DESIGN.md for choices not determined by the paper.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

NFFT, HOP, RATE = 4096, 1024, 44100
BINS = NFFT // 2 + 1
SOURCES = ('vocals', 'bass', 'drums', 'other')

class CumulativeNorm(nn.Module):
    """Causal group normalization across channels and frequency, accumulating time."""
    def __init__(self, channels, groups=1, eps=1e-5):
        super().__init__()
        self.groups, self.eps = groups, eps
        self.scale = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        b,c,t,f = x.shape
        z = x.reshape(b,self.groups,c//self.groups,t,f)
        # Float32 moments even in mixed precision; prefix statistics see no future.
        v = z.float()
        n = torch.arange(1,t+1,device=x.device,dtype=torch.float32) * (c//self.groups*f)
        avg = v.sum(dim=(2,4)).cumsum(dim=-1)/n
        sq = v.square().sum(dim=(2,4)).cumsum(dim=-1)/n
        y = (v-avg[:,:,None,:,None])*torch.rsqrt((sq-avg.square()).clamp_min(0)[:,:,None,:,None]+self.eps)
        return y.reshape(b,c,t,f).to(x.dtype)*self.scale[None,:,None,None]+self.bias[None,:,None,None]

class ChannelNorm(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm=nn.LayerNorm(c)
    def forward(self,x): return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)

class CausalConv(nn.Module):
    def __init__(self,ci,co,kt=3,kf=3,groups=1):
        super().__init__(); self.kt,self.kf=kt,kf
        self.conv=nn.Conv2d(ci,co,(kt,kf),groups=groups)
    def forward(self,x):
        return self.conv(F.pad(x,(self.kf//2,(self.kf-1)//2,self.kt-1,0)))

class ConvModule(nn.Module):
    """Bottleneck temporal convolution per frequency bin, residual connection."""
    def __init__(self,c):
        super().__init__(); h=c//4
        self.norm=CumulativeNorm(c)
        self.expand=nn.Conv2d(c,2*h,1)
        self.depthwise=CausalConv(h,h,3,1,groups=h)
        self.norm2=CumulativeNorm(h)
        self.project=nn.Conv2d(h,c,1)
    def forward(self,x):
        y=F.glu(self.expand(self.norm(x)),dim=1)
        return x+self.project(F.silu(self.norm2(self.depthwise(y))))

def bands(f):
    a=int(f*.175); b=int(f*.392)
    return (a,b,f-a-b)

class SparseDown(nn.Module):
    def __init__(self,ci,co,f):
        super().__init__(); self.lengths=bands(f); self.strides=(1,4,16); self.kernels=(3,4,16)
        self.pads=[]; self.out_lengths=[]
        self.down=nn.ModuleList(); self.blocks=nn.ModuleList()
        for n,s,k,d in zip(self.lengths,self.strides,self.kernels,(3,2,1)):
            total=(math.ceil(n/s)-1)*s+k-n
            self.pads.append((total//2,total-total//2))
            self.out_lengths.append(math.ceil(n/s))
            self.down.append(nn.Conv2d(ci,co,(1,k),stride=(1,s)))
            self.blocks.append(nn.Sequential(*[ConvModule(co) for _ in range(d)]))
        self.mix=CausalConv(co,co)
    def forward(self,x):
        parts=torch.split(x,self.lengths,dim=-1)
        ys=[]
        for p,conv,mods,(l,r) in zip(parts,self.down,self.blocks,self.pads):
            ys.append(mods(F.gelu(conv(F.pad(p,(l,r,0,0))))))
        skip=torch.cat(ys,-1)
        return self.mix(skip),skip

class SparseUp(nn.Module):
    def __init__(self,ci,co,down):
        super().__init__(); self.lengths=down.out_lengths; self.pads=down.pads; self.targets=down.lengths
        self.up=nn.ModuleList([nn.ConvTranspose2d(ci,co,(1,k),stride=(1,s)) for s,k in zip(down.strides,down.kernels)])
    def forward(self,x):
        return torch.cat([conv(p)[...,l:l+n] for p,conv,(l,r),n in zip(torch.split(x,self.lengths,-1),self.up,self.pads,self.targets)],-1)

class FrequencyConv(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm=ChannelNorm(c); self.conv=nn.Conv2d(c,c,(1,3),padding=(0,1),groups=8); self.act=nn.PReLU(c)
    def forward(self,x): return self.act(self.conv(self.norm(x)))

class FullBand(nn.Module):
    def __init__(self,c,f,squeeze=8):
        super().__init__(); self.norm=ChannelNorm(c); self.down=nn.Conv2d(c,squeeze,1); self.up=nn.Conv2d(squeeze,c,1)
        self.weight=nn.Parameter(torch.empty(squeeze,f,f)); self.bias=nn.Parameter(torch.zeros(squeeze,1,f))
        for w in self.weight: nn.init.kaiming_uniform_(w,a=math.sqrt(5))
    def forward(self,x):
        y=F.silu(self.down(self.norm(x)))
        # independent frequency projection for each squeezed channel
        y=torch.matmul(y,self.weight.transpose(-1,-2))+self.bias
        return F.silu(self.up(y))

def attend(q,k,v):
    # Explicit matmul is portable for ONNX. Mask future frames, not frequencies.
    t=q.shape[-2]; idx=torch.arange(t,device=q.device)
    logits=(q @ k.transpose(-1,-2))*(q.shape[-1]**-.5)
    logits=logits.masked_fill(idx[None,:]>idx[:,None],float('-inf'))
    return logits.softmax(-1) @ v

class TemporalAttention(nn.Module):
    def __init__(self,c,heads=4,dropout=0.):
        super().__init__(); self.heads=heads; self.norm=ChannelNorm(c); self.qkv=nn.Conv2d(c,3*c,1); self.out=nn.Conv2d(c,c,1); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        b,c,t,f=x.shape; h=self.heads
        q,k,v=self.qkv(self.norm(x)).chunk(3,1)
        def pack(y): return y.reshape(b,h,c//h,t,f).permute(0,4,1,3,2)
        y=attend(pack(q),pack(k),pack(v)).permute(0,2,4,3,1).reshape(b,c,t,f)
        return self.drop(self.out(y))

class TemporalFFN(nn.Module):
    def __init__(self,c,hidden=256,dropout=0.):
        super().__init__(); self.norm=ChannelNorm(c); self.up=nn.Conv2d(c,hidden,1); self.down=nn.Conv2d(hidden,c,1)
        self.convs=nn.ModuleList([CausalConv(hidden,hidden,5,1,groups=8) for _ in range(3)])
        self.gn=CumulativeNorm(hidden,8); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        x=F.silu(self.up(self.norm(x)))
        for i,conv in enumerate(self.convs):
            x=conv(x)
            if i==1: x=self.gn(x)
            x=F.silu(x)
        return self.drop(F.silu(self.down(x)))

class SeparationPair(nn.Module):
    def __init__(self,c,f,dropout=0.):
        super().__init__(); self.f1=FrequencyConv(c); self.full=FullBand(c,f); self.f2=FrequencyConv(c); self.attn=TemporalAttention(c,dropout=dropout); self.ffn=TemporalFFN(c,dropout=dropout)
    def forward(self,x):
        for block in (self.f1,self.full,self.f2,self.attn,self.ffn): x=x+block(x)
        return x

def projection(ci,co): return nn.Sequential(nn.Conv2d(ci,co,1),nn.PReLU(co),ChannelNorm(co))

class CSAFusion(nn.Module):
    def __init__(self,c,f,compress_gate=False,heads=4,embedding=1024):
        super().__init__(); self.h=heads; self.d=math.ceil(embedding/f)
        self.q=projection(c,self.d*heads); self.k=projection(c,self.d*heads); self.v=projection(c,c); self.out=projection(c,c)
        self.compress=compress_gate; self.pad=f%2
        if compress_gate:
            self.gate_down=nn.Conv2d(c,c,(1,2),stride=(1,2)); self.gate_up=nn.ConvTranspose2d(c,c,(1,2),stride=(1,2))
        gf=(f+1)//2 if compress_gate else f
        self.gate_norm=ChannelNorm(c); self.gate_linear=nn.Linear(gf,gf)
        self.glu=CausalConv(c,2*c,3,3); self.final=nn.Conv2d(c,c,1)
    def forward(self,x,skip):
        x=x+skip; b,c,t,f=x.shape; h=self.h
        def pack(y): return y.reshape(b,h,-1,t,f).permute(0,1,3,2,4).flatten(3)
        y=attend(pack(self.q(x)),pack(self.k(x)),pack(self.v(x)))
        y=y.reshape(b,h,t,c//h,f).permute(0,1,3,2,4).reshape(b,c,t,f)
        y=self.out(y)
        g=self.gate_down(F.pad(x,(0,self.pad))) if self.compress else x
        g=self.gate_linear(self.gate_norm(g)).sigmoid()
        if self.compress: g=self.gate_up(g)[...,:f]
        return self.final(F.glu(self.glu(y*g),dim=1))

class BandSCNet(nn.Module):
    def __init__(self,layers=4,dropout=0.):
        super().__init__(); dims=(4,32,64,128); f=BINS
        self.encoder=nn.ModuleList(); self.frequencies=[f]
        for ci,co in zip(dims,dims[1:]):
            layer=SparseDown(ci,co,f); self.encoder.append(layer); f=sum(layer.out_lengths); self.frequencies.append(f)
        self.separation=nn.Sequential(*[SeparationPair(dims[-1],f,dropout) for _ in range(layers)])
        self.fusions=nn.ModuleList(); self.decoder=nn.ModuleList()
        for i in (2,1,0):
            self.fusions.append(CSAFusion(dims[i+1],self.frequencies[i+1],compress_gate=i==0))
            self.decoder.append(SparseUp(dims[i+1],dims[i] if i else 16,self.encoder[i]))
    def forward(self,x):
        skips=[]
        for enc in self.encoder:
            x,s=enc(x); skips.append(s)
        x=self.separation(x)
        for fuse,up,skip in zip(self.fusions,self.decoder,reversed(skips)):
            x=up(fuse(x,skip))
        return x

# Audio transforms intentionally outside the export graph. Rectangular window follows SCNet.
def encode(wave):
    b,c,n=wave.shape
    padded=max(NFFT, NFFT+math.ceil(max(0,n-NFFT)/HOP)*HOP)
    wave=F.pad(wave,(0,padded-n)).reshape(b*c,padded)
    z=torch.stft(wave,NFFT,HOP,window=torch.ones(NFFT,device=wave.device,dtype=wave.dtype),center=False,normalized=True,return_complex=True)
    t=z.shape[-1]
    return torch.view_as_real(z).reshape(b,c,BINS,t,2).permute(0,1,4,3,2).reshape(b,c*2,t,BINS)

def decode(spec,length):
    b,features,t,f=spec.shape
    z=spec.reshape(b,features//2,2,t,f).permute(0,1,4,3,2).contiguous()
    z=torch.view_as_complex(z).reshape(-1,f,t)
    wave=torch.istft(z,NFFT,HOP,window=torch.ones(NFFT,device=spec.device,dtype=spec.dtype),center=False,normalized=True,length=NFFT+(t-1)*HOP)
    return wave.reshape(b,features//4,2,-1)[...,:length]
