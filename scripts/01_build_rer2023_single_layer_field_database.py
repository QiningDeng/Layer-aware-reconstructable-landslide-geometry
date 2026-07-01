#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a single-layer field-first SDF/corner/mask database from landslide polygons in a shapefile."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None



def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def largest_polygon(geom):
    if geom is None or geom.is_empty: return None
    try: geom=geom.buffer(0)
    except Exception: pass
    if geom.is_empty: return None
    if isinstance(geom, Polygon): return geom
    if isinstance(geom, MultiPolygon):
        parts=[g for g in geom.geoms if isinstance(g,Polygon) and not g.is_empty]
        return max(parts, key=lambda g:g.area) if parts else None
    try:
        u=unary_union(geom)
        if isinstance(u,Polygon): return u
        if isinstance(u,MultiPolygon):
            parts=[g for g in u.geoms if isinstance(g,Polygon) and not g.is_empty]
            return max(parts, key=lambda g:g.area) if parts else None
    except Exception: pass
    return None

def polygon_descriptors(poly:Polygon)->Dict[str,float]:
    area=float(poly.area); per=float(poly.length)
    minx,miny,maxx,maxy=poly.bounds
    bw=max(maxx-minx,1e-9); bh=max(maxy-miny,1e-9); barea=bw*bh
    aspect=max(bw,bh)/max(min(bw,bh),1e-9)
    compact=4*math.pi*area/max(per*per,1e-9)
    extent=area/max(barea,1e-9)
    hull=poly.convex_hull; harea=float(hull.area) if hull and not hull.is_empty else area
    solidity=area/max(harea,1e-9)
    pts=np.asarray(poly.exterior.coords, dtype=np.float64)
    vcnt=max(0,len(pts)-1); holes=len(poly.interiors)
    if vcnt>=4:
        p=pts[:-1]; v1=p-np.roll(p,1,axis=0); v2=np.roll(p,-1,axis=0)-p
        n1=np.linalg.norm(v1,axis=1)+1e-9; n2=np.linalg.norm(v2,axis=1)+1e-9
        ca=np.clip(np.sum(v1*v2,axis=1)/(n1*n2),-1,1); ang=np.arccos(ca)
        high=float(np.mean(ang<np.deg2rad(135))); mean_turn=float(np.mean(np.pi-ang)); max_turn=float(np.max(np.pi-ang))
    else: high=mean_turn=max_turn=0.0
    return dict(area=area,perimeter=per,bbox_width=bw,bbox_height=bh,bbox_area=barea,aspect_ratio=aspect,compactness=compact,extent=extent,solidity=solidity,convexity_defect=1-solidity,vertex_count=float(vcnt),hole_count=float(holes),high_turn_rate=high,mean_turn=mean_turn,max_turn=max_turn)

def assign_abcd(d):
    if d['aspect_ratio']>=5.0: return 'B_slender'
    if d['hole_count']>=1 or d['solidity']<0.72 or (d['vertex_count']>=120 and d['solidity']<0.86): return 'D_complex'
    if d['high_turn_rate']>=0.22 or d['compactness']<0.18: return 'C_high_curvature'
    return 'A_normal'

def sample_stratified(df,total,class_col,seed):
    rng=np.random.default_rng(seed); cnt=df[class_col].value_counts().sort_index(); ratio=cnt/cnt.sum()
    raw=ratio*total; base=np.floor(raw).astype(int); rem=total-int(base.sum())
    frac=(raw-base).sort_values(ascending=False)
    for c in frac.index[:rem]: base[c]+=1
    parts=[]
    for c,n in base.items():
        g=df[df[class_col]==c]
        if len(g)==0 or n<=0: continue
        parts.append(df.loc[rng.choice(g.index.to_numpy(), size=int(n), replace=len(g)<int(n))])
    out=pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out.iloc[:total].copy()

def split_stratified(df,class_col,seed):
    rng=np.random.default_rng(seed); sp={"train":[],"val":[],"test":[]}
    for cls,g in df.groupby(class_col):
        names=g['sample_name'].to_numpy(); rng.shuffle(names); n=len(names)
        nt=int(round(n*0.6)); nv=int(round(n*0.2));
        if nt+nv>n: nv=max(0,n-nt)
        if n>=3:
            nt=max(1,nt); nv=max(1,nv); ntest=n-nt-nv
            if ntest==0: ntest=1; nt=max(1,nt-1)
        sp['train']+=names[:nt].tolist(); sp['val']+=names[nt:nt+nv].tolist(); sp['test']+=names[nt+nv:].tolist()
    for k in sp: rng.shuffle(sp[k])
    return sp

def normalize_poly(poly,res,pad):
    minx,miny,maxx,maxy=poly.bounds; w=max(maxx-minx,1e-9); h=max(maxy-miny,1e-9); scale=(res-2*pad)/max(w,h)
    def tr(coords):
        a=np.asarray(coords,dtype=np.float64); x=(a[:,0]-minx)*scale+pad; y=(maxy-a[:,1])*scale+pad; return np.stack([x,y],axis=1)
    p=Polygon(tr(poly.exterior.coords), [tr(r.coords) for r in poly.interiors])
    try: p=p.buffer(0)
    except Exception: pass
    meta=dict(orig_minx=float(minx),orig_miny=float(miny),orig_maxx=float(maxx),orig_maxy=float(maxy),scale_to_pixel=float(scale),padding=int(pad),resolution=int(res))
    return p,meta

def poly_to_mask(poly,res):
    m=np.zeros((res,res),np.uint8)
    polys=list(poly.geoms) if isinstance(poly,MultiPolygon) else [poly]
    for p in polys:
        if p is None or p.is_empty: continue
        ext=np.round(np.asarray(p.exterior.coords,dtype=np.float32)).astype(np.int32).reshape(-1,1,2); cv2.fillPoly(m,[ext],255)
        for ring in p.interiors:
            pts=np.round(np.asarray(ring.coords,dtype=np.float32)).astype(np.int32).reshape(-1,1,2); cv2.fillPoly(m,[pts],0)
    return (m>0).astype(np.uint8)

def post_mask(m):
    m=(m>0).astype(np.uint8); n,lab,stat,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        keep=1+int(np.argmax(stat[1:,cv2.CC_STAT_AREA])); m=(lab==keep).astype(np.uint8)
    flood=m.copy(); h,w=m.shape; ff=np.zeros((h+2,w+2),np.uint8); cv2.floodFill(flood,ff,(0,0),1)
    holes=(flood==0).astype(np.uint8); return np.maximum(m,holes).astype(np.uint8)

def sdf_from_mask(mask,clip):
    inside=(mask>0).astype(np.uint8); outside=1-inside
    di=cv2.distanceTransform(inside,cv2.DIST_L2,5); do=cv2.distanceTransform(outside,cv2.DIST_L2,5)
    sdf=do-di # inside negative, outside positive
    return np.clip(sdf/max(clip,1e-6),-1,1).astype(np.float32)

def corner_field(poly,res,sigma,angle_deg):
    out=np.zeros((res,res),np.float32)
    pts=np.asarray(poly.exterior.coords[:-1],np.float32) if poly and not poly.is_empty else np.zeros((0,2),np.float32)
    if len(pts)<4: return out
    prev=np.roll(pts,1,axis=0); nxt=np.roll(pts,-1,axis=0); v1=pts-prev; v2=nxt-pts
    n1=np.linalg.norm(v1,axis=1)+1e-9; n2=np.linalg.norm(v2,axis=1)+1e-9
    ang=np.arccos(np.clip(np.sum(v1*v2,axis=1)/(n1*n2),-1,1)); turn=np.pi-ang
    cps=pts[turn>=np.deg2rad(angle_deg)]
    imp=np.zeros_like(out)
    for x,y in cps:
        ix=int(round(x)); iy=int(round(y))
        if 0<=ix<res and 0<=iy<res: imp[iy,ix]=1
    k=int(max(3,math.ceil(sigma*6)//2*2+1)); blur=cv2.GaussianBlur(imp,(k,k),sigmaX=sigma,sigmaY=sigma)
    return (blur/blur.max()).astype(np.float32) if blur.max()>0 else blur.astype(np.float32)

def save_preview(path,mask,sdf,corner):
    h=mask.shape[0]; can=np.zeros((h,h*3,3),np.uint8)
    can[:,:h]=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    can[:,h:2*h]=cv2.applyColorMap(((sdf+1)*127.5).clip(0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    can[:,2*h:3*h]=cv2.applyColorMap((corner*255).clip(0,255).astype(np.uint8),cv2.COLORMAP_HOT)
    cv2.imwrite(str(path),can)


def _gui_root():
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter 不可用，请在命令行中显式提供 --shp 和 --output。")
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def select_input_shp() -> str:
    root = _gui_root()
    path = filedialog.askopenfilename(
        title="选择输入 shapefile（*.shp）",
        filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def select_output_folder() -> str:
    root = _gui_root()
    folder = filedialog.askdirectory(title="选择数据库输出文件夹（可在对话框中新建）")
    root.destroy()
    return folder

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shp', default=''); ap.add_argument('--output', default='')
    ap.add_argument('--total-samples', type=int, default=0, help='样本总数。0 或负数表示使用 shp 中全部有效 polygon 样本。'); ap.add_argument('--resolution',type=int,default=256); ap.add_argument('--padding',type=int,default=12)
    ap.add_argument('--sdf-clip',type=float,default=32); ap.add_argument('--corner-sigma',type=float,default=2.0); ap.add_argument('--corner-angle-threshold',type=float,default=35)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--preview-count',type=int,default=48)
    ap.add_argument('--id-column',default=''); ap.add_argument('--class-column',default='')
    args=ap.parse_args()
    if not args.shp:
        args.shp = select_input_shp()
    if not args.output:
        args.output = select_output_folder()
    if not args.shp:
        raise SystemExit("未选择输入 shapefile。")
    if not args.output:
        raise SystemExit("未选择输出文件夹。")
    seed_all(args.seed)
    root=ensure_dir(args.output); feat=ensure_dir(root/'features'); meta=ensure_dir(root/'metadata'); rep=ensure_dir(root/'reports'); prevd=ensure_dir(root/'previews')
    print('[Read]',args.shp); gdf=gpd.read_file(args.shp)
    rows=[]; geoms=[]
    for i,geom in tqdm(list(enumerate(gdf.geometry)), total=len(gdf), desc='inspect'):
        poly=largest_polygon(geom)
        if poly is None or poly.area<=0: continue
        d=polygon_descriptors(poly); cls=str(gdf.loc[i,args.class_column]) if args.class_column and args.class_column in gdf.columns else assign_abcd(d)
        sid=str(gdf.loc[i,args.id_column]) if args.id_column and args.id_column in gdf.columns else str(i)
        rows.append(dict(source_index=int(i),source_id=sid,abcd_class=cls,**d)); geoms.append(poly)
    desc=pd.DataFrame(rows); desc['geom_list_index']=np.arange(len(desc))
    target_total = len(desc) if int(args.total_samples) <= 0 else min(int(args.total_samples), len(desc))
    print(f"[Sampling] valid polygons={len(desc)}, requested_total_samples={args.total_samples}, target_total={target_total}")
    if target_total >= len(desc):
        selected = desc.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        selected = sample_stratified(desc, target_total, 'abcd_class', args.seed).reset_index(drop=True)
    selected['sample_index']=np.arange(len(selected)); selected['sample_name']=[f'sample_{i:06d}' for i in range(len(selected))]
    print('[Selected]'); print(selected['abcd_class'].value_counts().sort_index().to_string())
    feat_rows=[]; self_rows=[]
    for _,r in tqdm(selected.iterrows(), total=len(selected), desc='build'):
        name=r.sample_name; poly0=geoms[int(r.geom_list_index)]; polyn,tmeta=normalize_poly(poly0,args.resolution,args.padding)
        mask=post_mask(poly_to_mask(polyn,args.resolution)); sdf=sdf_from_mask(mask,args.sdf_clip); corner=corner_field(polyn,args.resolution,args.corner_sigma,args.corner_angle_threshold)
        feature=np.stack([sdf,corner,mask.astype(np.float32)],0).astype(np.float32)
        np.savez_compressed(feat/f'{name}_features.npz', feature_tensor=feature, sdf_normalized=sdf[None].astype(np.float32), corner_field=corner[None].astype(np.float32), mask=mask[None].astype(np.float32), layer_names=np.array(['landslide'],dtype=object), vertical_ranks=np.array(['single'],dtype=object), source_image=np.array(str(r.source_id),dtype=object), sample_name=np.array(name,dtype=object), source_index=np.array(int(r.source_index),dtype=np.int64), abcd_class=np.array(str(r.abcd_class),dtype=object))
        rec=(sdf<=0).astype(np.uint8); inter=np.logical_and(rec,mask).sum(); union=np.logical_or(rec,mask).sum(); iou=float(inter/union) if union else 1.0
        fr=r.to_dict(); fr.update(tmeta); fr['feature_file']=str(feat/f'{name}_features.npz'); feat_rows.append(fr); self_rows.append(dict(sample_name=name,selfcheck_iou=iou))
        if int(r.sample_index)<args.preview_count: save_preview(prevd/f'{name}_preview.png',mask,sdf,corner)
    fdf=pd.DataFrame(feat_rows); fdf.to_csv(meta/'shape_descriptors.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(self_rows).to_csv(meta/'selfcheck_metrics.csv',index=False,encoding='utf-8-sig')
    splits=split_stratified(fdf,'abcd_class',args.seed); (meta/'split_stratified.json').write_text(json.dumps({'splits':splits},indent=2,ensure_ascii=False),encoding='utf-8')
    fdf[['sample_name','source_index','source_id','abcd_class','feature_file']].to_csv(meta/'selected_indices.csv',index=False,encoding='utf-8-sig')
    manifest=dict(type='single_layer_field_first_database',source_shapefile=str(args.shp),output_root=str(root),total_samples=int(len(fdf)),resolution=int(args.resolution),channels=['sdf_normalized','corner_field','mask'],train_val_test_ratio='6:2:2',split_counts={k:len(v) for k,v in splits.items()},class_counts=fdf['abcd_class'].value_counts().sort_index().to_dict(),sdf_sign='inside <= 0, outside > 0')
    (rep/'database_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
