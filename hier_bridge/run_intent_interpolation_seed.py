"""Transductive-content intent interpolation for strict pure cold recommendation.

Cold content may define the intent vocabulary, but cold interactions never enter
training. Image/text use their complete original features and separate models.
"""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'acrg_apr_github'))
from run_mmrec_seq_scl import Config,build_samples,leakage_checks,read_interactions,temporal_item_split

def l2_blocks(raw,block=256):
    out=np.empty(raw.shape,np.float32)
    for s in range(0,len(raw),block):
        x=np.asarray(raw[s:s+block],np.float32);out[s:s+len(x)]=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
    return out
def softmax_rows(x,t=.12):
    y=x/t;y-=y.max(1,keepdims=True);y=np.exp(y);return (y/y.sum(1,keepdims=True)).astype(np.float32)
def intent_space(x,k,seed):
    # Fit on every item's content, never on cold interactions (transductive content).
    rng=np.random.default_rng(seed);best=None
    for _ in range(5):
        centers=x[rng.choice(len(x),k,False)].copy()
        previous=None
        for __ in range(60):
            lab=np.argmax(x@centers.T,1);new=np.zeros_like(centers)
            for j in range(k):
                rows=x[lab==j];new[j]=rows.mean(0) if len(rows) else x[rng.integers(len(x))]
            centers=new/np.maximum(np.linalg.norm(new,axis=1,keepdims=True),1e-9)
            if previous is not None and np.array_equal(lab,previous):break
            previous=lab
        sim=x@centers.T;objective=float(np.max(sim,1).sum())
        if best is None or objective>best[0]:best=(objective,centers.copy(),lab.copy())
    objective,centers,lab=best;q=softmax_rows(x@centers.T);return q,centers,float(len(x)-objective),np.bincount(lab,minlength=k)
def sequences(inter,warm,cutoff,excluded):
    users={}
    for u,i,t in inter:
        if u not in excluded and i in warm and t<cutoff:users.setdefault(u,[]).append((t,i))
    return [sorted(v,key=lambda r:(r[0],r[1])) for v in users.values() if len(v)>1]
def transition(rows,q,k,smooth=.5):
    count=np.full((k,k),smooth,np.float64);edges=0
    for seq in rows:
        for (ta,a),(tb,b) in zip(seq,seq[1:]):
            if ta<tb:count+=np.outer(q[a],q[b]);edges+=1
    return (count/count.sum(1,keepdims=True)).astype(np.float32),edges
def distributions(prefix,q,A,delta=.85):
    w=delta**np.arange(len(prefix)-1,-1,-1);w=w/w.sum();current=(q[np.asarray(prefix)]*w[:,None]).sum(0);future=current@A;future/=max(future.sum(),1e-9);return current.astype(np.float32),future.astype(np.float32)
def z(x):return (x-x.mean())/max(float(x.std()),1e-6)
def cache(samples,candidates,image,text,qi,qt,Ai,At):
    cand=np.asarray(candidates);out=[]
    for s in samples:
        ci,fi=distributions(s.prefix,qi,Ai);ct,ft=distributions(s.prefix,qt,At)
        # Intent interpolation and original-feature local similarity are retained separately.
        ii_cur=qi[cand]@ci;ii_fut=qi[cand]@fi;it_cur=qt[cand]@ct;it_fut=qt[cand]@ft
        hi=np.asarray(s.prefix);local_i=np.max(image[hi]@image[cand].T,0);local_t=np.max(text[hi]@text[cand].T,0)
        pos=int(np.flatnonzero(cand==s.target)[0]);out.append((s,pos,z(ii_cur),z(ii_fut),z(local_i),z(it_cur),z(it_fut),z(local_t)))
    return out
def score(row,alpha,g_i,g_t,eta_i,eta_t):
    _,_,ic,iff,di,tc,tf,dt=row
    # convex current/future intent query; local term explicitly protects near-history items
    si=eta_i*((1-g_i)*ic+g_i*iff)+(1-eta_i)*di
    st=eta_t*((1-g_t)*tc+g_t*tf)+(1-eta_t)*dt
    return alpha*si+(1-alpha)*st
def evaluate(rows,cfg,pred=False):
    ranks=[];ps=[]
    for rid,r in enumerate(rows):
        sc=score(r,**cfg);target=float(sc[r[1]]);rank=int(1+np.count_nonzero(sc>target));ranks.append(rank)
        if pred:ps.append({'row_id':rid,'user':r[0].user,'target':r[0].target,'rank':rank,'history_length':len(r[0].prefix),'target_score':float(sc[r[1]])})
    m={}
    for k in (10,20):m[f'Recall@{k}']=float(np.mean(np.asarray(ranks)<=k));m[f'NDCG@{k}']=float(np.mean([1/math.log2(x+1) if x<=k else 0 for x in ranks]));m[f'MRR@{k}']=float(np.mean([1/x if x<=k else 0 for x in ranks]))
    return m,ps
def write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protocol(inter,warm,cutoff,excluded,qi,qt,k):
    seq=sequences(inter,warm,cutoff,excluded);Ai,ei=transition(seq,qi,k);At,et=transition(seq,qt,k);return Ai,At,{'users':len(seq),'image_edges':ei,'text_edges':et}
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2022);p.add_argument('--intents',type=int,default=32);p.add_argument('--validation-users',type=int);p.add_argument('--test-users',type=int);p.add_argument('--output-dir',type=Path,default=HERE/'results/baby_intent_interpolation_seed2022');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    root=ROOT/'baby';inter=read_interactions(root,'baby');warm,val,test,cutoff,_=temporal_item_split(inter,Config(max_sequence_length=10));train,valid,tests,_=build_samples(inter,warm,val,test,cutoff,10);checks=leakage_checks(train,valid,tests,warm,val,test)
    if a.validation_users:valid=valid[:a.validation_users]
    if a.test_users:tests=tests[:a.test_users]
    start=time.perf_counter();image=l2_blocks(np.load(root/'image_feat.npy',mmap_mode='r'));text=l2_blocks(np.load(root/'text_feat.npy',mmap_mode='r'));qi,_,ini,bini=intent_space(image,a.intents,a.seed+1);qt,_,int_,bint=intent_space(text,a.intents,a.seed+2)
    Aiv,Atv,dv=protocol(inter,warm,cutoff,{s.user for s in valid},qi,qt,a.intents);Ait,Att,dtg=protocol(inter,warm,cutoff,{s.user for s in tests},qi,qt,a.intents);prep=time.perf_counter()-start
    vc=cache(valid,sorted(val),image,text,qi,qt,Aiv,Atv);grid=[]
    # eta=1 means intent-only; eta=0 means original-feature local-only.
    for alpha in (0.,.1,.2,.3,.5):
      for gi in (0.,.25,.5,.75):
       for gt in (0.,.25,.5,.75):
        for eta in (.25,.5,.75,1.):
         cfg={'alpha':alpha,'g_i':gi,'g_t':gt,'eta_i':eta,'eta_t':eta};m,_=evaluate(vc,cfg);grid.append({**cfg,**m})
    best=max(grid,key=lambda x:(x['NDCG@10'],x['Recall@10']));keys=('alpha','g_i','g_t','eta_i','eta_t');locked={k:best[k] for k in keys};write(a.output_dir/'validation_grid.csv',grid);(a.output_dir/'locked_config.json').write_text(json.dumps(locked,indent=2))
    tc=cache(tests,sorted(test),image,text,qi,qt,Ait,Att);m,preds=evaluate(tc,locked,True);write(a.output_dir/'predictions.csv',preds)
    variants=[('Direct original features',{'alpha':locked['alpha'],'g_i':0,'g_t':0,'eta_i':0,'eta_t':0}),('Current intent only',{'alpha':locked['alpha'],'g_i':0,'g_t':0,'eta_i':1,'eta_t':1}),('Future intent only',{'alpha':locked['alpha'],'g_i':1,'g_t':1,'eta_i':1,'eta_t':1}),('No future',dict(locked,g_i=0,g_t=0)),('Image branch',dict(locked,alpha=1)),('Text branch',dict(locked,alpha=0)),('Proposed',locked)]
    ab=[]
    for name,cfg in variants:ab.append({'model':name,**cfg,**evaluate(tc,cfg)[0]})
    write(a.output_dir/'ablation.csv',ab);result={'seed':a.seed,'protocol':'transductive content clustering; strict cold-interaction exclusion; full cold catalog','original_features':'image4096 text384, L2 only','intents_per_modality':a.intents,'train_samples':len(train),'validation_users':len(valid),'test_users':len(tests),'locked':locked,'test':m,'validation_graph':dv,'test_graph':dtg,'image_cluster_min_max':[int(bini.min()),int(bini.max())],'text_cluster_min_max':[int(bint.min()),int(bint.max())],'image_inertia':ini,'text_inertia':int_,'leakage_checks':checks,'preprocessing_seconds':prep};(a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
