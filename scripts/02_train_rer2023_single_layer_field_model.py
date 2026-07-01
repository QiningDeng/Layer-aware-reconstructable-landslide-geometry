#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/evaluate single-layer field-first VAE or auto-decoder on SDF/corner/mask database."""
from __future__ import annotations
import argparse, json, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None

from typing import Any, Dict, List, Optional, Sequence, Tuple
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def _gui_root():
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter 不可用，请在命令行中显式提供路径参数。")
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def select_feature_root() -> str:
    root = _gui_root()
    folder = filedialog.askdirectory(title="选择已构建的 single-layer field 数据库文件夹")
    root.destroy()
    return folder


def select_output_root() -> str:
    root = _gui_root()
    folder = filedialog.askdirectory(title="选择训练/评价输出文件夹（可在对话框中新建）")
    root.destroy()
    return folder


def select_checkpoint_file() -> str:
    root = _gui_root()
    path = filedialog.askopenfilename(
        title="选择模型 checkpoint（best.pt 或 latest.pt）",
        filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def resolve_gui_paths(args):
    if args.command == "train":
        if not getattr(args, "feature_root", ""):
            args.feature_root = select_feature_root()
        if not getattr(args, "output_root", ""):
            args.output_root = select_output_root()
        if not args.feature_root:
            raise SystemExit("未选择数据库文件夹。")
        if not args.output_root:
            raise SystemExit("未选择训练输出文件夹。")
    elif args.command == "evaluate":
        if not getattr(args, "checkpoint", ""):
            args.checkpoint = select_checkpoint_file()
        if not args.checkpoint:
            raise SystemExit("未选择 checkpoint 文件。")
        # feature_root/output_root 可为空；为空时优先使用 checkpoint 内记录或 checkpoint 上级目录。
    return args

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def ensure_dir(p): p=Path(p); p.mkdir(parents=True,exist_ok=True); return p
def write_json(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def append_jsonl(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); open(p,'a',encoding='utf-8').write(json.dumps(o,ensure_ascii=False)+'\n')

@dataclass
class Config:
    feature_root:str; output_root:str; model_type:str='autodecoder'; seed:int=2026; latent_dim:int=64; encoder_input_size:int=128; eval_grid_size:int=256; batch_size:int=16; epochs:int=5000; learning_rate:float=5e-5; latent_learning_rate:float=5e-3; weight_decay:float=1e-6; points_per_sample:int=8192; eval_chunk_size:int=32768; validate_every:int=1; patience:int=50; min_delta:float=1e-5; save_every:int=10; num_workers:int=4; beta_kl:float=1e-4; weight_sdf:float=1.0; weight_corner:float=0.15; weight_mask:float=0.5; weight_dice:float=0.5; weight_consistency:float=0.05; weight_latent_reg:float=1e-4; boundary_delta:float=0.10; boundary_boost:float=4.0; mask_temperature:float=0.075; decoder_hidden_dim:int=256; decoder_num_layers:int=6; decoder_fourier_levels:int=8; posterior_steps:int=300; posterior_lr:float=2e-2; val_posterior_steps:int=100; val_posterior_lr:float=2e-2; val_max_batches:int=0; show_val_progress:bool=True; cache_in_memory:bool=False; device:str='cuda'

def load_splits(root):
    obj=json.loads((Path(root)/'metadata'/'split_stratified.json').read_text(encoding='utf-8')); return obj['splits'] if 'splits' in obj else obj

class FeatureDataset(Dataset):
    def __init__(self,root,split,enc_size,latent_indices=None,cache=False,max_samples=0):
        self.root=Path(root); self.enc_size=int(enc_size); self.latent_indices=latent_indices or {}; self.cache=cache; self._cache={}
        names=list(load_splits(root)[split]);
        if max_samples and max_samples>0: names=names[:max_samples]
        self.names=names; self.files=[self.root/'features'/f'{n}_features.npz' for n in names]
        desc=self.root/'metadata'/'shape_descriptors.csv'; self.class_map={}
        if desc.exists():
            df=pd.read_csv(desc)
            if 'sample_name' in df.columns and 'abcd_class' in df.columns: self.class_map=dict(zip(df.sample_name.astype(str),df.abcd_class.astype(str)))
        if cache:
            for f in tqdm(self.files,desc=f'cache {split}',leave=False): self._cache[str(f)]=self._load(f)
    def __len__(self): return len(self.files)
    def _load(self,path):
        path=Path(path)
        with np.load(path,allow_pickle=True) as d:
            ft=d['feature_tensor'].astype(np.float32); sdf=d['sdf_normalized'].astype(np.float32); corner=d['corner_field'].astype(np.float32); mask=d['mask'].astype(np.float32)
            name=str(d['sample_name'].tolist()) if 'sample_name' in d else path.stem.replace('_features',''); src=str(d['source_image'].tolist()) if 'source_image' in d else name
        small=cv2.resize(np.transpose(ft,(1,2,0)),(self.enc_size,self.enc_size),interpolation=cv2.INTER_LINEAR); small=np.transpose(small,(2,0,1)).astype(np.float32)
        return dict(sample_name=name,source_image=src,feature_small=small,sdf=sdf,corner=corner,mask=mask,latent_index=int(self.latent_indices.get(name,-1)),abcd_class=self.class_map.get(name,'unknown'))
    def __getitem__(self,i):
        k=str(self.files[i]); return self._cache[k] if k in self._cache else self._load(k)

def collate(batch):
    return dict(sample_name=[b['sample_name'] for b in batch], source_image=[b['source_image'] for b in batch], abcd_class=[b['abcd_class'] for b in batch], feature_small=torch.from_numpy(np.stack([b['feature_small'] for b in batch])), sdf=torch.from_numpy(np.stack([b['sdf'] for b in batch])), corner=torch.from_numpy(np.stack([b['corner'] for b in batch])), mask=torch.from_numpy(np.stack([b['mask'] for b in batch])), latent_indices=torch.tensor([b['latent_index'] for b in batch],dtype=torch.long))

class Fourier(nn.Module):
    def __init__(self,n=8): super().__init__(); self.n=int(n); self.register_buffer('freqs',2.0**torch.arange(self.n,dtype=torch.float32),persistent=False)
    @property
    def output_dim(self): return 2+4*self.n
    def forward(self,c):
        x=c[...,None]*self.freqs[None,None,None,:]*math.pi
        return torch.cat([c,torch.sin(x).reshape(c.shape[0],c.shape[1],-1),torch.cos(x).reshape(c.shape[0],c.shape[1],-1)],-1)
class ConvEncoder(nn.Module):
    def __init__(self,in_ch=3,latent_dim=64):
        super().__init__(); ch=[32,64,128,256,256]; layers=[]; ci=in_ch
        for co in ch: layers += [nn.Conv2d(ci,co,4,2,1),nn.BatchNorm2d(co),nn.LeakyReLU(0.2,True)]; ci=co
        self.features=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool2d((1,1)); self.mu=nn.Linear(ch[-1],latent_dim); self.lv=nn.Linear(ch[-1],latent_dim)
    def forward(self,x): h=self.pool(self.features(x)).flatten(1); return self.mu(h),self.lv(h)
class Decoder(nn.Module):
    def __init__(self,latent_dim,hidden,layers_n,fourier):
        super().__init__(); self.ff=Fourier(fourier); d=self.ff.output_dim+latent_dim; layers=[]
        for _ in range(layers_n-1): layers += [nn.Linear(d,hidden),nn.SiLU(True)]; d=hidden
        layers.append(nn.Linear(d,3)); self.net=nn.Sequential(*layers)
    def forward(self,z,coords):
        B,P,_=coords.shape; cf=self.ff(coords); ze=z[:,None,:].expand(-1,P,-1); y=self.net(torch.cat([cf,ze],-1))
        return {'sdf':y[...,0:1],'corner_logits':y[...,1:2],'mask_logits':y[...,2:3]}
class VAE(nn.Module):
    def __init__(self,cfg): super().__init__(); self.encoder=ConvEncoder(3,cfg.latent_dim); self.decoder=Decoder(cfg.latent_dim,cfg.decoder_hidden_dim,cfg.decoder_num_layers,cfg.decoder_fourier_levels)
class AutoDecoder(nn.Module):
    def __init__(self,n_train,cfg): super().__init__(); self.latent=nn.Embedding(n_train,cfg.latent_dim); nn.init.normal_(self.latent.weight,0,0.01); self.decoder=Decoder(cfg.latent_dim,cfg.decoder_hidden_dim,cfg.decoder_num_layers,cfg.decoder_fourier_levels)

def grid(h,w,device):
    ys=torch.linspace(-1,1,h,device=device); xs=torch.linspace(-1,1,w,device=device); yy,xx=torch.meshgrid(ys,xs,indexing='ij'); return torch.stack([xx,yy],-1)
def sample_points(sdf,corner,mask,coords_flat,npts):
    B,_,H,W=sdf.shape; HW=H*W; idx=torch.randint(0,HW,(B,npts),device=sdf.device)
    def g(x): return torch.gather(x.view(B,1,HW).permute(0,2,1),1,idx.unsqueeze(-1))
    return {'coords':coords_flat[idx],'sdf':g(sdf),'corner':g(corner),'mask':g(mask)}
def dice_loss(logits,target,eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*target).sum(1); union=p.sum(1)+target.sum(1); return 1-((2*inter+eps)/(union+eps)).mean()
def loss_total(out,gsdf,gcorner,gmask,cfg,z=None,mu=None,lv=None):
    w=1+cfg.boundary_boost*torch.exp(-torch.abs(gsdf)/max(cfg.boundary_delta,1e-6)); lsdf=torch.mean(w*F.smooth_l1_loss(out['sdf'],gsdf,reduction='none'))
    lc=F.mse_loss(torch.sigmoid(out['corner_logits']),gcorner); lm=F.binary_cross_entropy_with_logits(out['mask_logits'],gmask); ld=dice_loss(out['mask_logits'],gmask)
    lcon=F.mse_loss(torch.sigmoid(out['mask_logits']),torch.sigmoid(-out['sdf']/max(cfg.mask_temperature,1e-6)))
    total=cfg.weight_sdf*lsdf+cfg.weight_corner*lc+cfg.weight_mask*lm+cfg.weight_dice*ld+cfg.weight_consistency*lcon
    lkl=torch.tensor(0.,device=gsdf.device); lreg=torch.tensor(0.,device=gsdf.device)
    if mu is not None and lv is not None: lkl=0.5*torch.mean(torch.sum(torch.exp(lv)+mu**2-1-lv,1)); total=total+cfg.beta_kl*lkl
    if z is not None: lreg=torch.mean(z**2); total=total+cfg.weight_latent_reg*lreg
    return {'total':total,'sdf':lsdf,'corner':lc,'mask_bce':lm,'dice':ld,'consistency':lcon,'kl':lkl,'latent_reg':lreg}
def reparam(mu,lv): return mu+torch.randn_like(mu)*torch.exp(0.5*lv)

def train_epoch(model,loader,opt,cfg,device):
    model.train(); keys=['total','sdf','corner','mask_bce','dice','consistency','kl','latent_reg']; vals={k:[] for k in keys}
    H=int(next(iter(loader))['sdf'].shape[-1]); coords=grid(H,H,device).view(-1,2)
    for b in tqdm(loader,desc='train',leave=False):
        sdf=b['sdf'].to(device); corner=b['corner'].to(device); mask=b['mask'].to(device); smp=sample_points(sdf,corner,mask,coords,cfg.points_per_sample); opt.zero_grad(set_to_none=True)
        if cfg.model_type=='vae':
            mu,lv=model.encoder(b['feature_small'].to(device)); z=reparam(mu,lv); out=model.decoder(z,smp['coords']); L=loss_total(out,smp['sdf'],smp['corner'],smp['mask'],cfg,mu=mu,lv=lv)
        else:
            z=model.latent(b['latent_indices'].to(device)); out=model.decoder(z,smp['coords']); L=loss_total(out,smp['sdf'],smp['corner'],smp['mask'],cfg,z=z)
        L['total'].backward(); opt.step()
        for k in keys: vals[k].append(float(L[k].detach().cpu()))
    return {f'train_{k}_loss':float(np.mean(v)) for k,v in vals.items() if v}
def val_loss(model,loader,cfg,device):
    """
    Validation loss with visible progress bar.

    For VAE:
        use encoder mu as deterministic latent.

    For auto-decoder:
        optimize a temporary latent z for each validation batch for
        cfg.val_posterior_steps steps, then compute validation loss.
    """
    model.eval()
    keys=['total','sdf','corner','mask_bce','dice','consistency','kl','latent_reg']
    vals={k:[] for k in keys}

    first=next(iter(loader))
    H=int(first['sdf'].shape[-1])
    coords=grid(H,H,device).view(-1,2)

    max_batches=int(getattr(cfg,'val_max_batches',0) or 0)
    total_batches=len(loader) if max_batches<=0 else min(len(loader),max_batches)
    show_bar=bool(getattr(cfg,'show_val_progress',True))

    val_iter=tqdm(
        enumerate(loader),
        total=total_batches,
        desc=f"val posterior({int(getattr(cfg,'val_posterior_steps',0))} steps)",
        leave=False,
        dynamic_ncols=True,
        disable=(not show_bar)
    )

    for bi,b in val_iter:
        if max_batches>0 and bi>=max_batches:
            break

        sdf=b['sdf'].to(device)
        corner=b['corner'].to(device)
        mask=b['mask'].to(device)

        if cfg.model_type=='vae':
            with torch.no_grad():
                smp=sample_points(sdf,corner,mask,coords,cfg.points_per_sample)
                mu,lv=model.encoder(b['feature_small'].to(device))
                out=model.decoder(mu,smp['coords'])
                L=loss_total(out,smp['sdf'],smp['corner'],smp['mask'],cfg,mu=mu,lv=lv)

        else:
            z=torch.zeros((sdf.shape[0],cfg.latent_dim),device=device,requires_grad=True)
            optz=torch.optim.Adam([z],lr=float(getattr(cfg,'val_posterior_lr',cfg.posterior_lr)))
            steps=int(getattr(cfg,'val_posterior_steps',100))

            model.decoder.eval()
            decoder_params=list(model.decoder.parameters())
            old_requires_grad=[p.requires_grad for p in decoder_params]
            for p in decoder_params:
                p.requires_grad_(False)

            running_inner_loss=float("nan")
            try:
                inner_iter=range(steps)
                # 只显示 batch 级进度条，避免每个 batch 都刷 100 行；但在 batch 进度条 postfix 中显示内部 posterior loss。
                for si in inner_iter:
                    smp=sample_points(sdf,corner,mask,coords,cfg.points_per_sample)
                    optz.zero_grad(set_to_none=True)
                    outz=model.decoder(z,smp['coords'])
                    Lz=loss_total(outz,smp['sdf'],smp['corner'],smp['mask'],cfg,z=z)['total']
                    Lz.backward()
                    optz.step()
                    if si==0 or si==steps-1 or ((si+1) % max(1, steps//5) == 0):
                        running_inner_loss=float(Lz.detach().cpu())
                        if show_bar:
                            val_iter.set_postfix({
                                "batch": f"{bi+1}/{total_batches}",
                                "z_step": f"{si+1}/{steps}",
                                "z_loss": f"{running_inner_loss:.4f}"
                            })
            finally:
                for p,req in zip(decoder_params,old_requires_grad):
                    p.requires_grad_(req)

            with torch.no_grad():
                smp=sample_points(sdf,corner,mask,coords,cfg.points_per_sample)
                out=model.decoder(z.detach(),smp['coords'])
                L=loss_total(out,smp['sdf'],smp['corner'],smp['mask'],cfg,z=z.detach())

        current={}
        for k in keys:
            v=float(L[k].detach().cpu())
            vals[k].append(v)
            current[k]=v

        if show_bar:
            mean_total=float(np.mean(vals['total'])) if vals['total'] else float("nan")
            val_iter.set_postfix({
                "batch": f"{bi+1}/{total_batches}",
                "val_loss": f"{mean_total:.4f}"
            })

    return {f'val_{k}_loss':float(np.mean(v)) for k,v in vals.items() if v}

@torch.no_grad()
def dense(decoder,z,gs,device,chunk):
    decoder.eval(); coords=grid(gs,gs,device).view(-1,2); ss=[]; cc=[]; mm=[]
    for s in range(0,coords.shape[0],chunk):
        out=decoder(z,coords[s:s+chunk][None]); ss.append(out['sdf'].squeeze(0).cpu()); cc.append(torch.sigmoid(out['corner_logits']).squeeze(0).cpu()); mm.append(torch.sigmoid(out['mask_logits']).squeeze(0).cpu())
    return {'sdf':torch.cat(ss).view(gs,gs).numpy().astype(np.float32),'corner':torch.cat(cc).view(gs,gs).numpy().astype(np.float32),'mask_prob':torch.cat(mm).view(gs,gs).numpy().astype(np.float32)}
def post(mask):
    m=(mask>0).astype(np.uint8); n,lab,stat,_=cv2.connectedComponentsWithStats(m,8); out=np.zeros_like(m)
    if n>1:
        keep=1+int(np.argmax(stat[1:,cv2.CC_STAT_AREA])); out=(lab==keep).astype(np.uint8)
    if out.any():
        flood=out.copy(); h,w=out.shape; ff=np.zeros((h+2,w+2),np.uint8); cv2.floodFill(flood,ff,(0,0),1); out=np.maximum(out,(flood==0).astype(np.uint8))
    return out

def boundary(m):
    cs,_=cv2.findContours((m>0).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=[c.reshape(-1,2).astype(np.float32) for c in cs if len(c)>1]
    return np.concatenate(pts,0) if pts else np.zeros((0,2),np.float32)
def chamfer(a,b):
    if len(a)==0 and len(b)==0: return 0.0
    if len(a)==0 or len(b)==0: return float('inf')
    def mm(x,y):
        vals=[]
        for s in range(0,len(x),2048):
            d=x[s:s+2048,None,:]-y[None,:,:]; vals.append(np.sqrt((d*d).sum(2)).min(1))
        return float(np.concatenate(vals).mean())
    return 0.5*(mm(a,b)+mm(b,a))
def haus(a,b):
    if len(a)==0 and len(b)==0: return 0.0
    if len(a)==0 or len(b)==0: return float('inf')
    def mx(x,y):
        vals=[]
        for s in range(0,len(x),2048):
            d=x[s:s+2048,None,:]-y[None,:,:]; vals.append(np.sqrt((d*d).sum(2)).min(1).max())
        return float(np.max(vals))
    return max(mx(a,b),mx(b,a))
def topo(m):
    m=(m>0).astype(np.uint8); n,_,_,_=cv2.connectedComponentsWithStats(m,8); comp=max(0,n-1); cs,h=cv2.findContours(m*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); holes=0
    if h is not None:
        holes=sum(1 for x in h[0] if x[3]!=-1)
    return comp,holes
def metrics(pred,gt):
    pred=(pred>0).astype(np.uint8); gt=(gt>0).astype(np.uint8); inter=float(np.logical_and(pred,gt).sum()); union=float(np.logical_or(pred,gt).sum()); pa=float(pred.sum()); ga=float(gt.sum()); iou=inter/union if union else 1.0; dice=2*inter/(pa+ga) if pa+ga else 1.0
    H,W=gt.shape; diag=math.hypot(H,W); bp,bg=boundary(pred),boundary(gt); ch,ha=chamfer(bp,bg),haus(bp,bg); comp,holes=topo(pred)
    def cen(m):
        y,x=np.where(m>0); return None if len(x)==0 else np.array([x.mean(),y.mean()])
    cp,cg=cen(pred),cen(gt); cd=1.0 if cp is None or cg is None else min(1.0,float(np.linalg.norm(cp-cg)/diag))
    return {'iou':iou,'dice':dice,'area_relative_error':abs(pa-ga)/ga if ga else 0.0,'centroid_distance_normalized':cd,'boundary_chamfer_normalized':1.0 if not np.isfinite(ch) else min(1.0,ch/diag),'boundary_hausdorff_normalized':1.0 if not np.isfinite(ha) else min(1.0,ha/diag),'component_count':float(comp),'hole_count':float(holes),'invalid_topology':float(comp!=1 or holes!=0)}
def visualize(path,gt,pred,title):
    gt=(gt>0).astype(np.uint8); pred=(pred>0).astype(np.uint8); H,W=gt.shape; ov=np.zeros((H,W,3),np.uint8); ov[np.logical_and(gt==1,pred==0)]=(0,180,180); ov[np.logical_and(gt==0,pred==1)]=(180,0,180); ov[np.logical_and(gt==1,pred==1)]=(255,220,0)
    fig,ax=plt.subplots(1,3,figsize=(15,5)); ax[0].imshow(gt,cmap='gray'); ax[0].set_title('GT'); ax[1].imshow(pred,cmap='gray'); ax[1].set_title('Field-first'); ax[2].imshow(ov); ax[2].set_title('Overlay')
    for a in ax: a.axis('off')
    fig.suptitle(title); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=180); plt.close(fig)
def opt_latent(decoder,sample,cfg,device):
    sdf=sample['sdf'].to(device); corner=sample['corner'].to(device); mask=sample['mask'].to(device); H=sdf.shape[-1]; coords=grid(H,H,device).view(-1,2); z=torch.zeros((1,cfg.latent_dim),device=device,requires_grad=True); opt=torch.optim.Adam([z],lr=cfg.posterior_lr)
    for _ in range(cfg.posterior_steps):
        smp=sample_points(sdf,corner,mask,coords,cfg.points_per_sample); opt.zero_grad(set_to_none=True); L=loss_total(decoder(z,smp['coords']),smp['sdf'],smp['corner'],smp['mask'],cfg,z=z)['total']; L.backward(); opt.step()
    return z.detach()

def build_model(cfg,n_train,device):
    if cfg.model_type=='vae':
        m=VAE(cfg).to(device); opt=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    else:
        m=AutoDecoder(n_train,cfg).to(device); opt=torch.optim.AdamW([{'params':m.decoder.parameters(),'lr':cfg.learning_rate},{'params':m.latent.parameters(),'lr':cfg.latent_learning_rate}],weight_decay=cfg.weight_decay)
    return m,opt

def save_ckpt(p,cfg,m,opt,epoch,best,best_epoch,pat,hist):
    torch.save({'config':asdict(cfg),'model_type':cfg.model_type,'epoch':epoch,'best_score':best,'best_epoch':best_epoch,'patience_count':pat,'model_state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict() if opt else None,'history':hist},p)
def load_ckpt(p,m,opt,device):
    ck=torch.load(p,map_location=device); m.load_state_dict(ck['model_state_dict'],strict=True)
    if opt is not None and ck.get('optimizer_state_dict') is not None:
        opt.load_state_dict(ck['optimizer_state_dict'])
        for st in opt.state.values():
            for k,v in list(st.items()):
                if torch.is_tensor(v): st[k]=v.to(device)
    return ck

def eval_split(m,split,cfg,save_visuals=64,max_samples=0):
    device=torch.device(cfg.device if torch.cuda.is_available() or cfg.device=='cpu' else 'cpu'); train_names=load_splits(cfg.feature_root)['train']; lidx={n:i for i,n in enumerate(train_names)}; ds=FeatureDataset(cfg.feature_root,split,cfg.encoder_input_size,lidx,cfg.cache_in_memory,max_samples=max_samples); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate); rows=[]; visuals=[]; m.eval()
    for b in tqdm(loader,desc=f'eval {split}'):
        name=b['sample_name'][0]; cls=b['abcd_class'][0]; gt=b['mask'][0,0].numpy().astype(np.uint8)
        if cfg.model_type=='vae':
            with torch.no_grad(): mu,_=m.encoder(b['feature_small'].to(device)); z=mu
        else:
            if split=='train' and name in lidx:
                with torch.no_grad(): z=m.latent(torch.tensor([lidx[name]],device=device))
            else: z=opt_latent(m.decoder,{'sdf':b['sdf'],'corner':b['corner'],'mask':b['mask']},cfg,device)
        d=dense(m.decoder,z,cfg.eval_grid_size,device,cfg.eval_chunk_size); pred=post((d['sdf']<=0).astype(np.uint8)); gt2=cv2.resize(gt,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST) if gt.shape!=pred.shape else gt; met=metrics(pred,gt2); row={'sample_name':name,'split':split,'abcd_class':cls,**met}; rows.append(row); visuals.append((met['iou'],name,gt2.copy(),pred.copy(),cls))
    root=ensure_dir(Path(cfg.output_root)/'evaluation'/f'{split}_best'); df=pd.DataFrame(rows); df.to_csv(root/'sample_metrics.csv',index=False,encoding='utf-8-sig')
    summ={'split':split,'samples':int(len(df))}
    for c in [c for c in df.columns if c not in {'sample_name','split','abcd_class'}]:
        summ[f'{c}_mean']=float(df[c].mean()); summ[f'{c}_p50']=float(df[c].median()); summ[f'{c}_p10']=float(df[c].quantile(0.1)); summ[f'{c}_min']=float(df[c].min())
    for cls,g in df.groupby('abcd_class'):
        summ[f'{cls}_count']=int(len(g)); summ[f'{cls}_iou_mean']=float(g['iou'].mean()); summ[f'{cls}_iou_p10']=float(g['iou'].quantile(0.1))
    write_json(root/'summary.json',summ)
    if save_visuals>0 and visuals:
        visuals.sort(key=lambda x:x[0]); sel=(visuals[:min(16,len(visuals))]+visuals[-min(16,len(visuals)):]); step=max(1,len(visuals)//max(1,save_visuals-len(sel))) if save_visuals>len(sel) else 1; sel=(sel+visuals[::step])[:save_visuals]; vdir=ensure_dir(root/'visuals')
        for r,(iou,name,gt,pred,cls) in enumerate(sel): visualize(vdir/f'{r:03d}_{name}_{cls}_iou{iou:.4f}.png',gt,pred,f'{split} | {name} | {cls} | IoU={iou:.4f}')
    return summ

def plot_hist(hist,out):
    if not hist: return
    df=pd.DataFrame(hist); ensure_dir(Path(out)/'curves'); df.to_csv(Path(out)/'history'/'training_history.csv',index=False,encoding='utf-8-sig')
    for cols,f,t in [(['train_total_loss','val_total_loss'],'loss_total.png','Total loss'),(['train_sdf_loss','train_mask_bce_loss','train_dice_loss','val_sdf_loss','val_mask_bce_loss','val_dice_loss'],'loss_components.png','Components')]:
        cols=[c for c in cols if c in df.columns]
        if not cols: continue
        plt.figure(figsize=(8,5)); [plt.plot(df['epoch'],df[c],label=c) for c in cols]; plt.title(t); plt.xlabel('Epoch'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(Path(out)/'curves'/f,dpi=180); plt.close()

def cmd_train(a):
    cfg=Config(feature_root=a.feature_root,output_root=a.output_root,model_type=a.model_type,seed=a.seed,latent_dim=a.latent_dim,encoder_input_size=a.encoder_input_size,eval_grid_size=a.eval_grid_size,batch_size=a.batch_size,epochs=a.epochs,learning_rate=a.learning_rate,latent_learning_rate=a.latent_learning_rate,points_per_sample=a.points_per_sample,validate_every=a.validate_every,patience=a.patience,min_delta=a.min_delta,save_every=a.save_every,num_workers=a.num_workers,posterior_steps=a.posterior_steps,posterior_lr=a.posterior_lr,val_posterior_steps=a.val_posterior_steps,val_posterior_lr=a.val_posterior_lr,val_max_batches=a.val_max_batches,show_val_progress=(not a.no_val_progress),cache_in_memory=a.cache_in_memory,device=a.device)
    seed_all(cfg.seed); out=ensure_dir(cfg.output_root); cdir=ensure_dir(out/'checkpoints'); hdir=ensure_dir(out/'history'); write_json(out/'config.json',asdict(cfg)); device=torch.device(cfg.device if torch.cuda.is_available() or cfg.device=='cpu' else 'cpu')
    names=load_splits(cfg.feature_root)['train']; lidx={n:i for i,n in enumerate(names)}; trds=FeatureDataset(cfg.feature_root,'train',cfg.encoder_input_size,lidx,cfg.cache_in_memory); vds=FeatureDataset(cfg.feature_root,'val',cfg.encoder_input_size,lidx,cfg.cache_in_memory)
    tr=DataLoader(trds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=device.type=='cuda',collate_fn=collate); va=DataLoader(vds,batch_size=cfg.batch_size,shuffle=False,num_workers=max(0,min(2,cfg.num_workers)),pin_memory=device.type=='cuda',collate_fn=collate)
    m,opt=build_model(cfg,len(names),device); start=1; best=float('inf'); be=0; pat=0; hist=[]
    if a.resume:
        ck=load_ckpt(a.resume,m,opt,device); start=int(ck.get('epoch',0))+1; best=float(ck.get('best_score',best)); be=int(ck.get('best_epoch',0)); pat=int(ck.get('patience_count',0)); hist=list(ck.get('history',[]));
        if getattr(a,'reset_monitor_on_resume',False):
            best=float('inf'); be=0; pat=0
            print('[Resume] monitor reset: best=inf, best_epoch=0, patience=0')
        print('[Resume]',a.resume,'start',start)
    print(f'[Data] train={len(trds)}, val={len(vds)}, model={cfg.model_type}, device={device}')
    for ep in range(start,cfg.epochs+1):
        t=time.time(); tl=train_epoch(m,tr,opt,cfg,device); row={'epoch':ep,**tl}
        if ep%cfg.validate_every==0: vl=val_loss(m,va,cfg,device); row.update(vl); score=float(vl.get('val_total_loss',tl['train_total_loss']))
        else: score=float(tl['train_total_loss'])
        if score<best-cfg.min_delta: best=score; be=ep; pat=0; improved=True
        else: pat+=1; improved=False
        row.update(score=score,best_score=best,best_epoch=be,patience=pat,elapsed_seconds=time.time()-t); hist.append(row); append_jsonl(hdir/'training_history.jsonl',row); pd.DataFrame(hist).to_csv(hdir/'training_history.csv',index=False,encoding='utf-8-sig')
        save_ckpt(cdir/'latest.pt',cfg,m,opt,ep,best,be,pat,hist)
        if improved: save_ckpt(cdir/'best.pt',cfg,m,opt,ep,best,be,pat,hist)
        if ep%cfg.save_every==0: save_ckpt(cdir/f'epoch_{ep:06d}.pt',cfg,m,opt,ep,best,be,pat,hist)
        plot_hist(hist,out); print(f"[Epoch {ep:06d}] train={tl['train_total_loss']:.6f} val={row.get('val_total_loss',float('nan')):.6f} best={best:.6f}@{be} patience={pat}/{cfg.patience} time={row['elapsed_seconds']:.1f}s")
        if pat>=cfg.patience: print('[Early stopping]'); break
    ck=load_ckpt(cdir/'best.pt',m,None,device); summ={}
    for sp,vis in [('train',a.final_train_visuals),('val',a.final_val_visuals),('test',a.final_test_visuals)]: summ[sp]=eval_split(m,sp,cfg,vis,a.final_max_samples)
    ensure_dir(out/'evaluation'); write_json(out/'evaluation'/'best_train_val_test_summary.json',summ); pd.DataFrame([{'split':k,**v} for k,v in summ.items()]).to_csv(out/'evaluation'/'best_train_val_test_summary.csv',index=False,encoding='utf-8-sig'); print('[Final]',out/'evaluation'/'best_train_val_test_summary.csv')

def cmd_eval(a):
    tmp=torch.load(a.checkpoint,map_location='cpu'); cfg=Config(**tmp['config']); cfg.feature_root=a.feature_root or cfg.feature_root; cfg.output_root=a.output_root or str(Path(a.checkpoint).parent.parent); cfg.device=a.device; cfg.posterior_steps=a.posterior_steps; cfg.posterior_lr=a.posterior_lr
    device=torch.device(cfg.device if torch.cuda.is_available() or cfg.device=='cpu' else 'cpu'); names=load_splits(cfg.feature_root)['train']; m,_=build_model(cfg,len(names),device); load_ckpt(a.checkpoint,m,None,device); print(json.dumps(eval_split(m,a.split,cfg,a.save_visuals,a.max_samples),indent=2,ensure_ascii=False))

def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True); tr=sub.add_parser('train')
    for q,req,default in [('feature-root',False,''),('output-root',False,'')]: tr.add_argument('--'+q,required=req,default=default)
    tr.add_argument('--model-type',choices=['vae','autodecoder'],default='autodecoder'); tr.add_argument('--resume',default=''); tr.add_argument('--seed',type=int,default=2026); tr.add_argument('--latent-dim',type=int,default=64); tr.add_argument('--encoder-input-size',type=int,default=128); tr.add_argument('--eval-grid-size',type=int,default=256); tr.add_argument('--batch-size',type=int,default=16); tr.add_argument('--epochs',type=int,default=5000); tr.add_argument('--learning-rate',type=float,default=5e-5); tr.add_argument('--latent-learning-rate',type=float,default=5e-3); tr.add_argument('--points-per-sample',type=int,default=8192); tr.add_argument('--validate-every',type=int,default=1); tr.add_argument('--patience',type=int,default=50); tr.add_argument('--min-delta',type=float,default=1e-5); tr.add_argument('--save-every',type=int,default=10); tr.add_argument('--num-workers',type=int,default=4); tr.add_argument('--posterior-steps',type=int,default=300); tr.add_argument('--posterior-lr',type=float,default=2e-2); tr.add_argument('--val-posterior-steps',type=int,default=100); tr.add_argument('--val-posterior-lr',type=float,default=2e-2); tr.add_argument('--val-max-batches',type=int,default=0); tr.add_argument('--no-val-progress',action='store_true'); tr.add_argument('--device',default='cuda'); tr.add_argument('--cache-in-memory',action='store_true'); tr.add_argument('--monitor-metric',choices=['auto','train_total_loss','val_total_loss'],default='auto'); tr.add_argument('--reset-monitor-on-resume',action='store_true'); tr.add_argument('--final-max-samples',type=int,default=0); tr.add_argument('--final-train-visuals',type=int,default=0); tr.add_argument('--final-val-visuals',type=int,default=64); tr.add_argument('--final-test-visuals',type=int,default=64)
    ev=sub.add_parser('evaluate'); ev.add_argument('--checkpoint',default=''); ev.add_argument('--feature-root',default=''); ev.add_argument('--output-root',default=''); ev.add_argument('--split',choices=['train','val','test'],default='test'); ev.add_argument('--max-samples',type=int,default=0); ev.add_argument('--posterior-steps',type=int,default=300); ev.add_argument('--posterior-lr',type=float,default=2e-2); ev.add_argument('--save-visuals',type=int,default=64); ev.add_argument('--device',default='cuda')
    return p
if __name__=='__main__':
    a=resolve_gui_paths(parser().parse_args()); cmd_train(a) if a.command=='train' else cmd_eval(a)
