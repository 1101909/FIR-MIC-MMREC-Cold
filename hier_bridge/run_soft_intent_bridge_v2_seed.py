"""Behavior-supervised soft-intent sequential bridging, CPU pilot."""
from __future__ import annotations
import argparse,csv,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'acrg_apr_github'))
from run_intent_interpolation_seed import l2_blocks,intent_space
from run_mmrec_seq_scl import Config,Sample,build_samples,leakage_checks,read_interactions,temporal_item_split

class DS(Dataset):
 def __init__(self,x):self.x=x
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return self.x[i]
def collate(rows):
 l=torch.tensor([len(r.prefix) for r in rows]);w=int(l.max());p=torch.zeros(len(rows),w,dtype=torch.long)
 for j,r in enumerate(rows):p[j,:len(r.prefix)]=torch.tensor(r.prefix)
 return p,l,torch.tensor([r.target for r in rows]),rows
def pseudo_cold(samples,folds=5):
 out=[]
 for r in samples:
  f=r.target%folds;kept=[(i,t) for i,t in zip(r.prefix,r.prefix_times) if i%folds!=f]
  if kept:out.append(Sample(r.user,tuple(i for i,_ in kept),r.target,tuple(t for _,t in kept),r.target_time))
 return out
def warm_teacher(image,text,warm,k,seed,temp=.12):
 joint=np.concatenate([image/np.sqrt(2),text/np.sqrt(2)],1).astype(np.float32);wi=np.asarray(sorted(warm));qw,c,_,_=intent_space(joint[wi],k,seed);sim=joint@c.T;y=sim/temp;y-=y.max(1,keepdims=True);y=np.exp(y);q=(y/y.sum(1,keepdims=True)).astype(np.float32);return q,{'fit_items':'warm-only','intents':k,'min_warm_cluster':int(np.bincount(q[wi].argmax(1),minlength=k).min()),'max_warm_cluster':int(np.bincount(q[wi].argmax(1),minlength=k).max())}

class Model(nn.Module):
 def __init__(self,image,text,teacher,k,d=64,maxlen=10):
  super().__init__();self.register_buffer('image',image);self.register_buffer('text',text);self.register_buffer('teacher',teacher);self.ip=nn.Linear(image.shape[1],d);self.tp=nn.Linear(text.shape[1],d);self.gate=nn.Linear(3*d,2);self.proto_i=nn.Parameter(torch.randn(k,d)/math.sqrt(d));self.proto_t=nn.Parameter(torch.randn(k,d)/math.sqrt(d));self.intent_emb=nn.Parameter(torch.randn(k,d)/math.sqrt(d));self.pos=nn.Embedding(maxlen,d);layer=nn.TransformerEncoderLayer(d,4,2*d,.15,batch_first=True,norm_first=True);self.sas=nn.TransformerEncoder(layer,2);self.future=nn.Linear(d,k);self.k=k
 def projected(self,idx):return F.normalize(self.ip(self.image[idx]),dim=-1),F.normalize(self.tp(self.text[idx]),dim=-1)
 def item_q(self,idx):
  vi,tt=self.projected(idx);g=self.gate(torch.cat([tt,vi,tt*vi],-1)).softmax(-1);li=vi@F.normalize(self.proto_i,dim=-1).T;lt=tt@F.normalize(self.proto_t,dim=-1).T;return (g[...,0,None]*lt+g[...,1,None]*li).div(.10).softmax(-1)
 def content(self,idx):
  vi,tt=self.projected(idx);g=self.gate(torch.cat([tt,vi,tt*vi],-1)).softmax(-1);return F.normalize(g[...,0,None]*tt+g[...,1,None]*vi,dim=-1)
 def encode(self,p,l):
  b,w=p.shape;mask=torch.arange(w,device=p.device)[None]>=l[:,None];q=self.item_q(p);x=q@self.intent_emb+self.pos(torch.arange(w,device=p.device))[None];causal=torch.triu(torch.ones(w,w,device=p.device,dtype=torch.bool),1);h=self.sas(x,mask=causal,src_key_padding_mask=mask);last=h[torch.arange(b,device=p.device),l-1];future=self.future(last).softmax(-1);rec=.85**torch.arange(w-1,-1,-1,device=p.device).float();weights=rec[None]*(~mask);weights/=weights.sum(1,keepdims=True);current=(q*weights[:,:,None]).sum(1);return current,future,self.content(p),mask,q
 def components(self,p,l,candidates):
  cur,fut,hist,mask,_=self.encode(p,l);qc=self.item_q(candidates);cold=self.content(candidates);intent_res=(fut-cur)@qc.T;att=torch.einsum('bld,cd->blc',hist,cold).masked_fill(mask[:,:,None],-1e9).softmax(1);ctx=torch.einsum('blc,bld->bcd',att,hist);fine=torch.einsum('bcd,cd->bc',ctx,cold);return cur,fut,intent_res,fine

def raw_anchor(p,l,cand,image,text):
 vals=[]
 for j in range(len(p)):
  h=p[j,:l[j]];si=torch.max(image[h]@image[cand].T,0).values;st=torch.max(text[h]@text[cand].T,0).values;vals.append(.1*si+.9*st)
 return torch.stack(vals)
def train_model(model,samples,image,text,epochs,batch,seed,device):
 loader=DataLoader(DS(samples),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate);opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4);hist=[]
 for ep in range(1,epochs+1):
  model.train();tot=[];ints=[];ranks=[]
  for p,l,t,_ in loader:
   p=p.to(device);l=l.to(device);t=t.to(device);opt.zero_grad();cur,fut,ir,fine=model.components(p,l,t);qt=model.item_q(t);interest=-(qt.detach()*torch.log(fut+1e-9)).sum(1).mean();teacher=F.kl_div(torch.log(model.item_q(t)+1e-9),model.teacher[t],reduction='batchmean');anchor=raw_anchor(p,l,t,image,text);score=anchor+.1*ir+.1*(fine-anchor);labels=torch.arange(len(t),device=device);rank=F.cross_entropy(score/.07,labels);mean_q=model.item_q(t).mean(0);balance=(mean_q*torch.log(mean_q*model.k+1e-9)).sum();proto=torch.cat([F.normalize(model.proto_i,dim=-1),F.normalize(model.proto_t,dim=-1)],0);loss=rank+.25*interest+.05*teacher+.01*balance;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();tot.append(float(loss));ints.append(float(interest));ranks.append(float(rank))
  row={'epoch':ep,'loss':float(np.mean(tot)),'interest_loss':float(np.mean(ints)),'rank_loss':float(np.mean(ranks))};hist.append(row);print(json.dumps(row),flush=True)
 return hist
@torch.no_grad()
def make_cache(model,samples,candidates,image,text,batch,device):
 model.eval();cand=torch.tensor(candidates,device=device);lookup={x:i for i,x in enumerate(candidates)};out=[]
 for p,l,_,rows in DataLoader(DS(samples),batch_size=batch,collate_fn=collate):
  p=p.to(device);l=l.to(device);cur,fut,ir,fine=model.components(p,l,cand);anchor=raw_anchor(p,l,cand,image,text)
  for j,r in enumerate(rows):out.append((r,lookup[r.target],anchor[j].cpu().numpy(),ir[j].cpu().numpy(),(fine[j]-anchor[j]).cpu().numpy(),cur[j].cpu().numpy(),fut[j].cpu().numpy()))
 return out
def z(x):return (x-x.mean())/max(float(x.std()),1e-6)
def evaluate(rows,li,lf,pred=False):
 ranks=[];ps=[];ih1=[];ih3=[]
 for rid,(r,pos,a,ir,fr,cur,fut) in enumerate(rows):
  confidence=(1-(-np.sum(fut*np.log(fut+1e-9))/math.log(len(fut))))*(1-float(np.dot(cur,fut)/(max(np.linalg.norm(cur)*np.linalg.norm(fut),1e-9))));score=z(a)+li*confidence*z(ir)+lf*z(fr);target=float(score[pos]);rank=int(1+np.count_nonzero(score>target));ranks.append(rank)
  # cold target intent diagnostic uses predicted distribution vs learned target distribution approximated by candidate residual direction unavailable; save confidence.
  if pred:ps.append({'row_id':rid,'user':r.user,'target':r.target,'rank':rank,'history_length':len(r.prefix),'future_confidence':confidence,'target_score':target})
 m={}
 for k in (10,20):m[f'Recall@{k}']=float(np.mean(np.asarray(ranks)<=k));m[f'NDCG@{k}']=float(np.mean([1/math.log2(x+1) if x<=k else 0 for x in ranks]));m[f'MRR@{k}']=float(np.mean([1/x if x<=k else 0 for x in ranks]))
 return m,ps
def write(path,rows):
 with path.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2022);p.add_argument('--intents',type=int,default=64);p.add_argument('--dim',type=int,default=64);p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--validation-users',type=int);p.add_argument('--test-users',type=int);p.add_argument('--output-dir',type=Path,default=HERE/'results/baby_soft_intent_bridge_v2_seed2022');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 root=ROOT/'baby';inter=read_interactions(root,'baby');warm,val,test,cutoff,_=temporal_item_split(inter,Config(max_sequence_length=10));train,valid,tests,_=build_samples(inter,warm,val,test,cutoff,10);checks=leakage_checks(train,valid,tests,warm,val,test);ptrain=pseudo_cold(train)
 if a.validation_users:valid=valid[:a.validation_users]
 if a.test_users:tests=tests[:a.test_users]
 start=time.perf_counter();image_np=l2_blocks(np.load(root/'image_feat.npy',mmap_mode='r'));text_np=l2_blocks(np.load(root/'text_feat.npy',mmap_mode='r'));teacher,diag=warm_teacher(image_np,text_np,warm,a.intents,a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');image=torch.from_numpy(image_np).to(device);text=torch.from_numpy(text_np).to(device);model=Model(image,text,torch.from_numpy(teacher).to(device),a.intents,a.dim,10).to(device);prep=time.perf_counter()-start;ts=time.perf_counter();history=train_model(model,ptrain,image,text,a.epochs,a.batch_size,a.seed,device);train_seconds=time.perf_counter()-ts
 vc=make_cache(model,valid,sorted(val),image,text,a.batch_size,device);grid=[]
 for li in (0.,.025,.05,.1,.2):
  for lf in (0.,.025,.05,.1,.2):
   m,_=evaluate(vc,li,lf);grid.append({'lambda_intent':li,'lambda_fine':lf,**m})
 best=max(grid,key=lambda x:(x['NDCG@10'],x['Recall@10']));locked={'lambda_intent':best['lambda_intent'],'lambda_fine':best['lambda_fine']};write(a.output_dir/'validation_grid.csv',grid);write(a.output_dir/'training.csv',history);(a.output_dir/'locked_config.json').write_text(json.dumps(locked,indent=2));tc=make_cache(model,tests,sorted(test),image,text,a.batch_size,device);m,preds=evaluate(tc,locked['lambda_intent'],locked['lambda_fine'],True);write(a.output_dir/'predictions.csv',preds);ab=[]
 for name,li,lf in [('Direct anchor',0,0),('Anchor+future intent residual',locked['lambda_intent'],0),('Anchor+candidate decoder',0,locked['lambda_fine']),('Full',locked['lambda_intent'],locked['lambda_fine'])]:ab.append({'model':name,'lambda_intent':li,'lambda_fine':lf,**evaluate(tc,li,lf)[0]})
 write(a.output_dir/'ablation.csv',ab);result={'seed':a.seed,'protocol':'warm-only initialized behavior-supervised soft intents + SASRec + pseudo-cold prefixes + safe direct residual cold ranking','device':str(device),'original_features':'image4096 text384 learned full-input encoders','intents':a.intents,'dim':a.dim,'epochs':a.epochs,'train_samples_original':len(train),'train_samples_pseudo_cold':len(ptrain),'validation_users':len(valid),'test_users':len(tests),'intent_initialization':diag,'locked':locked,'test':m,'leakage_checks':checks,'preprocessing_seconds':prep,'train_seconds':train_seconds};(a.output_dir/'result.json').write_text(json.dumps(result,indent=2));torch.save({'state_dict':model.state_dict(),'result':result},a.output_dir/'model.pt');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
