"""Run unmodified official SEMCo and CLCRec model classes on MMREC-COLD.

Only data indexing and the common full-cold evaluator are project-owned. Model
layers and losses are imported from pinned GitHub submodules.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'hier_bridge'))
from run_mmrec_seq_scl import Config,read_interactions,temporal_item_split,build_samples
from run_intent_interpolation_seed import l2_blocks

def raw_sets(data,dataset):
 inter=read_interactions(data/dataset,dataset); warm,val,test,cutoff,_=temporal_item_split(inter,Config(max_sequence_length=10)); train,valid,tests,_=build_samples(inter,warm,val,test,cutoff,10)
 # Warm evidence available before the validation-cold boundary. This includes
 # test users' observed warm history, but never a cold interaction.
 train_pairs=sorted({(u,i) for u,i,t in inter if i in warm and t<cutoff})
 return inter,warm,val,test,train_pairs,valid,tests
def metrics(ranks):
 out={}; a=np.asarray(ranks)
 for k in (10,20):
  out[f'Recall@{k}']=float(np.mean(a<=k));out[f'NDCG@{k}']=float(np.mean([1/math.log2(r+1) if r<=k else 0 for r in ranks]));out[f'MRR@{k}']=float(np.mean([1/r if r<=k else 0 for r in ranks]))
 return out
def stable_rank(scores,target_position):
 # A deterministic total order is required when scores tie. Counting only
 # scores strictly greater than the target incorrectly makes every item rank 1
 # for an all-equal score vector (for example, a zero user representation).
 order=torch.argsort(scores,descending=True,stable=True)
 return int(torch.nonzero(order==target_position,as_tuple=False)[0,0].item()+1)
def rank_embeddings(user_emb,item_emb,user_map,item_map,samples,candidates):
 cpos=torch.tensor([item_map[i] for i in candidates],device=user_emb.device); ce=item_emb[cpos]; lookup={x:j for j,x in enumerate(candidates)}; ranks=[]
 for r in samples:
  s=user_emb[user_map[r.user]]@ce.T;ranks.append(stable_rank(s,lookup[r.target]))
 return metrics(ranks)

def rank_semco_prefixes(item_emb,item_map,samples,candidates):
 """Evaluate SEMCo with the exact leakage-safe prefix of each sample.

 SEMCo represents a user as normalized RY: the mean of the content embeddings
 of items in that user's observed history. Each temporal sample has its own R,
 so reusing one global pre-cutoff interaction row is not protocol-equivalent.
 """
 cpos=torch.tensor([item_map[i] for i in candidates],device=item_emb.device)
 candidate_emb=item_emb[cpos];lookup={item:j for j,item in enumerate(candidates)}
 ranks=[];zero_norm_users=0;all_equal_score_users=0;target_ties=[]
 for sample in samples:
  history_pos=torch.tensor([item_map[i] for i in sample.prefix],device=item_emb.device)
  user_raw=item_emb[history_pos].mean(dim=0)
  zero_norm_users+=int(torch.linalg.vector_norm(user_raw).item()<=1e-12)
  user_emb=F.normalize(user_raw,dim=0)
  scores=user_emb@candidate_emb.T;target_position=lookup[sample.target]
  target_score=scores[target_position]
  ties=int((scores==target_score).sum().item());target_ties.append(ties)
  all_equal_score_users+=int(ties==len(candidates))
  ranks.append(stable_rank(scores,target_position))
 diagnostics={
  'users':len(samples),'candidates':len(candidates),
  'zero_norm_users':zero_norm_users,'all_equal_score_users':all_equal_score_users,
  'mean_target_ties':float(np.mean(target_ties)) if target_ties else 0.0,
  'max_target_ties':max(target_ties,default=0),
 }
 return metrics(ranks),diagnostics

def patch_semco_compat(databuilder,evaluator):
 # The pinned commit is missing two symbols referenced by its own shared code.
 if not hasattr(databuilder,'DataBuilder'):
  class _UnusedWarmDataBuilder:
   def __init__(self,*args,**kwargs):
    raise RuntimeError('Warm-only DataBuilder is not shipped by this SEMCo commit')
  databuilder.DataBuilder=_UnusedWarmDataBuilder
 if not hasattr(evaluator.Metric,'hit_ratio'):
  @staticmethod
  def _hit_ratio(origin,hits):
   return round(sum(hits[user]>0 for user in origin)/len(origin),5)
  evaluator.Metric.hit_ratio=_hit_ratio
 if not getattr(evaluator.ranking_evaluation,'_empty_safe',False):
  upstream_ranking_evaluation=evaluator.ranking_evaluation
  def _ranking_evaluation(origin,res,N):
   if origin:
    return upstream_ranking_evaluation(origin,res,N)
   measure=[]
   for n in N:
    measure.extend([f'Top {n}\n','Hit Ratio:0.0\n','Recall:0.0\n','NDCG:0.0\n'])
   return measure,[[0.0,0.0] for _ in N]
  _ranking_evaluation._empty_safe=True
  evaluator.ranking_evaluation=_ranking_evaluation

def run_semco(data,dataset,epochs,batch,seed):
 sem=ROOT/'external/SEMCo';sys.path.insert(0,str(sem))
 # The pinned SEMCo commit exports only cold-start builders, while its shared
 # BaseRecommender imports the unused warm-only DataBuilder unconditionally.
 # Supply only that missing import symbol outside the upstream tree. SEMCo's
 # executed cold path still uses the official ColdStartDataBuilder unchanged.
 import util.databuilder as semco_databuilder
 # The pinned upstream evaluator calls Metric.hit_ratio(), but that method is
 # absent from the same commit. Restore ColdRec's user-level HR definition at
 # runtime so the official model code can complete its internal validation.
 import util.evaluator as semco_evaluator
 patch_semco_compat(semco_databuilder,semco_evaluator)
 from models.SEMCo import SEMCo
 _,warm,val,test,pairs,valid,tests=raw_sets(data,dataset)
 image=l2_blocks(np.load(data/dataset/'image_feat.npy',mmap_mode='r'));text=l2_blocks(np.load(data/dataset/'text_feat.npy',mmap_mode='r'))
 triple=lambda rows:[[int(u),int(i),1.] for u,i in rows]
 train=triple(pairs);v=triple((r.user,r.target) for r in valid);te=triple((r.user,r.target) for r in tests);empty=[]
 users=sorted({u for u,_ in pairs}|{r.user for r in valid}|{r.user for r in tests}); items=sorted(warm|val|test)
 # SEMCo's builder creates its source->mapped item table only from rows present
 # in train/validation/test lists, but full-catalog ranking also contains items
 # with no target interaction in these lists. Register those content-only items
 # under a synthetic metadata user in overall validation. These rows are never
 # used by training or by the common evaluator and therefore add no supervision.
 represented={i for _,i in pairs}|{r.target for r in valid}|{r.target for r in tests}
 registry_user=max(users)+1
 registry=[[registry_user,int(i),0.] for i in items if i not in represented]
 overall_v=v+registry
 args=SimpleNamespace(topN='10,20',model='SEMCo',dataset=dataset,emb_size=64,epochs=epochs,bs=batch,lr=.001,reg=.001,patience=10,decay_lr_epoch=[False,epochs],emb_sizes=(192,64),eval_batch_size=2048,sm_scale=12.,fn='sparsemax')
 model=SEMCo(args,train,empty,v,overall_v,empty,te,te,registry_user+1,len(items),users,sorted(warm),[],sorted(val|test),torch.device('cuda'),item_content=[image,text]);model.train()
 # Keep the official learned item encoder, but construct RY from each sample's
 # exact temporal prefix. The common evaluator restricts each split to its own
 # full cold candidate partition and applies deterministic stable tie-breaking.
 validation_metrics,validation_diag=rank_semco_prefixes(model.item_emb,model.data.item,valid,sorted(val))
 test_metrics,test_diag=rank_semco_prefixes(model.item_emb,model.data.item,tests,sorted(test))
 return {'validation':validation_metrics,'test':test_metrics,
         'diagnostics':{'validation':validation_diag,'test':test_diag}}

class PairDS(Dataset):
 def __init__(self,pairs,nuser,nitem,seen,neg,seed):self.pairs=pairs;self.nuser=nuser;self.nitem=nitem;self.seen=seen;self.neg=neg;self.r=random.Random(seed)
 def __len__(self):return len(self.pairs)
 def __getitem__(self,j):
  u,i=self.pairs[j]; pool=list(set(range(self.nitem))-self.seen[u]); ns=self.r.sample(pool,min(self.neg,len(pool)));ns+=(ns[:1]*(self.neg-len(ns)));return torch.tensor([u]*(self.neg+1)),torch.tensor([self.nuser+i]+[self.nuser+x for x in ns])
def run_clcrec(data,dataset,epochs,batch,seed):
 # Compatibility alias for the old upstream import; no model equation changes.
 import torch_geometric.utils as tgu
 if not hasattr(tgu,'scatter_'):tgu.scatter_=tgu.scatter
 clc=ROOT/'external/CLCRec';sys.path.insert(0,str(clc));from model_CLCRec import CLCRec
 _,warm,val,test,pairs,valid,tests=raw_sets(data,dataset); order=sorted(warm)+sorted(val)+sorted(test); imap={x:j for j,x in enumerate(order)};users=sorted({u for u,_ in pairs}|{r.user for r in valid}|{r.user for r in tests});umap={x:j for j,x in enumerate(users)}
 mapped=[(umap[u],imap[i]) for u,i in pairs];seen={u:set() for u in range(len(users))}
 for u,i in mapped:seen[u].add(i)
 image=torch.tensor(l2_blocks(np.load(data/dataset/'image_feat.npy',mmap_mode='r'))[order],dtype=torch.float,device='cuda');text=torch.tensor(l2_blocks(np.load(data/dataset/'text_feat.npy',mmap_mode='r'))[order],dtype=torch.float,device='cuda')
 neg=min(200,max(1,len(warm)-1)); model=CLCRec(len(users),len(order),len(warm),mapped,.1,64,image,None,text,.07,neg,.5,False,.5).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3);loader=DataLoader(PairDS(mapped,len(users),len(warm),seen,neg,seed),batch_size=batch,shuffle=True)
 model.train()
 for _ in range(epochs):
  for u,i in loader:opt.zero_grad();loss,_,_=model.loss(u.cuda(),i.cuda());loss.backward();opt.step()
 # Refresh official content-generated cold representations.
 with torch.no_grad():
  u,i=next(iter(loader));model(u.cuda(),i.cuda());ue=model.result[:len(users)];ie=model.result[len(users):]
 return {'validation':rank_embeddings(ue,ie,umap,imap,valid,sorted(val)),'test':rank_embeddings(ue,ie,umap,imap,tests,sorted(test))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--dataset',choices=['baby','clothing','sports'],required=True);p.add_argument('--models',nargs='+',choices=['semco','clcrec'],default=['semco','clcrec']);p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--seed',type=int,default=2022);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 out={};
 for name in a.models:out[name]=run_semco(a.data_dir,a.dataset,a.epochs,a.batch_size,a.seed) if name=='semco' else run_clcrec(a.data_dir,a.dataset,a.epochs,a.batch_size,a.seed)
 a.output.write_text(json.dumps({'dataset':a.dataset,'seed':a.seed,'epochs':a.epochs,'results':out},indent=2));print(json.dumps(out))
if __name__=='__main__':main()
