"""Physics-constrained residual trajectory Key-Value retrieval.

Keys are physical-edge residual history descriptors. Values preserve every
member node and every forecast lead of the same training episode. No forecast
error labels or validation/test targets enter the bank.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from backbone import PhysicalCFMemory as TemporalBackbone, graph_tensors, local_keys


def phase_values(model, origins, horizon):
    ids=origins[:,None]+torch.arange(horizon,device=origins.device)[None,:]
    return model.daily[ids%288]+model.weekly[ids%2016]


def history_phase(model, origins):
    ids=origins[:,None]+torch.arange(-model.seq_len,0,device=origins.device)[None,:]
    return model.daily[ids%288]+model.weekly[ids%2016]


def residual_history(raw, model, origins):
    return (raw-model.norm_mean-model.norm_std*history_phase(model,origins)) / model.norm_std


def residual_future(y, model, origins):
    return (y-model.norm_mean-model.norm_std*phase_values(model,origins,model.horizon)) / model.norm_std


class ResidualKVBank:
    def __init__(self,payload,device):
        self.origins=payload['origins'].to(device);self.values=payload['values'].to(device);self.squares=self.values.square()
        keys=payload['keys'].to(device);self.mean=keys.mean(0,keepdim=True);self.std=keys.std(0,unbiased=False,keepdim=True).clamp_min(1e-3);self.keys=(keys-self.mean)/self.std
        self.incidence=payload['incidence'].to(device);self.family=payload['family'].to(device);self.anchors=payload['anchors'].to(device)
    def eligibility(self,origins,seq_len,h):return self.origins[None,:]+h<=origins[:,None]-seq_len
    def retrieve(self,query,metric,origins,seq_len,h,topk=8,temp=.5):
        b,e,d=query.shape;w=F.softplus(metric)+.01;w=w/w.sum(-1,keepdim=True)
        q=(query-self.mean)/self.std
        dist=((q.square()*w).sum(-1).unsqueeze(-1)+(self.keys.square()*w).sum(-1).transpose(0,1)[None]-2*torch.einsum('bed,med,ed->bem',q,self.keys,w)).clamp_min(0)
        allowed=self.eligibility(origins,seq_len,h);dist=dist.masked_fill(~allowed[:,None,:],float('inf'))
        d,ids=dist.topk(min(topk,self.keys.size(0)),dim=-1,largest=False);valid=torch.isfinite(d);a=torch.softmax(-d.masked_fill(~valid,1e6)/temp,-1)*valid;a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
        values=self.values[ids];mean=(a[...,None,None]*values).sum(2);var=(a[...,None,None]*(values-mean[:, :, None]).square()).sum(2);support=valid.any(-1).to(query.dtype)
        return mean,var,support,d,ids,a


class PhysicsResidualKV(TemporalBackbone):
    def __init__(self,graph,mean,std,daily,weekly,seq_len=96,horizon=96,hidden=128,mode='physical',topk=8,temp=.5):
        super().__init__(graph,mean,std,daily,weekly,seq_len,horizon,hidden,bins=min(12,horizon));self.mode=mode;self.topk=topk;self.temp=temp
        self.register_buffer('edge_incidence',self.incidence.clone());self.register_buffer('edge_family',self.family_membership.clone());self.register_buffer('edge_anchors',self.anchors.clone())
        self.metric_kv=nn.Parameter(torch.zeros(self.family_membership.size(1),13));self.global_metric_kv=nn.Parameter(torch.zeros(1,graph['num_nodes']*3+2))
        self.edge_encoder_kv=nn.Sequential(nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.edge_attention_kv=nn.Sequential(nn.LayerNorm(hidden+3),nn.Linear(hidden+3,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.diff_encoder=nn.Sequential(nn.Linear(horizon,hidden),nn.GELU());self.unc_encoder=nn.Sequential(nn.Linear(horizon,hidden),nn.GELU())
        self.fusion_kv=nn.Sequential(nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,horizon));nn.init.zeros_(self.fusion_kv[-1].weight);nn.init.zeros_(self.fusion_kv[-1].bias)
        self.gate_kv=nn.Linear(hidden*3,1);nn.init.zeros_(self.gate_kv.weight);nn.init.zeros_(self.gate_kv.bias);self.direct_scale_kv=nn.Parameter(torch.zeros(horizon))
    def query_global(self,r,origins):
        last=r[:,-6:].mean(1);prev=r[:,-12:-6].mean(1);vol=r[:,-24:].std(1,unbiased=False);phase=origins.float()%288
        return torch.cat([last,last-prev,vol,torch.sin(2*math.pi*phase/288)[:,None],torch.cos(2*math.pi*phase/288)[:,None]],-1)[:,None]
    def forward(self,raw,origins,bank,disable_memory=False,uniform_edges=False):
        base,nodes=self.temporal_features(raw,origins);periodic=phase_values(self,origins,self.horizon)*self.norm_std+self.norm_mean
        rh=residual_history(raw,self,origins);pool=self.edge_incidence/self.edge_incidence.sum(0,keepdim=True).clamp_min(1)
        if self.mode=='global':
            q=self.query_global(rh,origins);w,var,support,dist,ids,alpha=bank.retrieve(q,self.global_metric_kv,origins,self.seq_len,self.horizon,self.topk,self.temp);node_w=alpha.expand(-1,self.num_nodes,-1);edge_att=None;node_support=support.expand(-1,self.num_nodes)
        else:
            q=local_keys(rh,origins,pool);metric=self.edge_family@self.metric_kv;w,var,support,dist,ids,alpha=bank.retrieve(q,metric,origins,self.seq_len,self.horizon,self.topk,self.temp)
            pooled=torch.einsum('bnd,ne->bed',nodes,pool);vv=(torch.einsum('bnd,ne->bed',nodes.square(),pool)-pooled.square()).clamp_min(0);meta=self.edge_family@self.family_embedding+self.size_embedding(self.edge_incidence.sum(0).long().clamp_max(32));edge=self.edge_encoder_kv(torch.cat([pooled+meta,nodes[:,self.edge_anchors],vv],-1));nearest=dist[:,:,:1];nearest=torch.where(torch.isfinite(nearest),nearest,torch.zeros_like(nearest));edge_att=torch.ones_like(support[:,:,None]) if uniform_edges else .001+.999*torch.sigmoid(self.edge_attention_kv(torch.cat([edge,torch.log1p(nearest),-(alpha*alpha.clamp_min(1e-8).log()).sum(-1,keepdim=True),support[:,:,None]],-1)));edge_att=edge_att*support[:,:,None];denom=torch.einsum('ne,be->bn',self.edge_incidence,edge_att.squeeze(-1));node_support=(denom>0).to(raw.dtype);node_w=torch.einsum('ne,bem->bnm',self.edge_incidence,alpha*edge_att);node_w=node_w/denom[:,:,None].clamp_min(1e-8)
        # Bank retrieval is edge-indexed; aggregate each edge's retrieved residual
        # values to nodes using the same incidence-normalized attention.
        edge_values=(alpha[...,None,None]*bank.values[ids]).sum(2)
        if self.mode=='global':mem=edge_values[:,0].permute(0,2,1).expand(-1,-1,self.num_nodes).transpose(1,2) if edge_values.dim()==4 else edge_values
        else:
            mem=torch.einsum('ne,behn->bhn',self.edge_incidence,edge_values*edge_att[:,:,:,None]);mem=mem/torch.einsum('ne,be->bn',self.edge_incidence,edge_att.squeeze(-1))[:,None].clamp_min(1e-8)
        # edge_values [B,E,H,N]; for global use node-specific original values.
        if self.mode=='global': mem=edge_values[:,0]
        residual_base=(base-periodic)/self.norm_std;diff=mem-residual_base;unc=(var.mean(1)+1e-6).sqrt();ctx=torch.cat([nodes,self.diff_encoder(diff.transpose(1,2)),self.unc_encoder(unc.transpose(1,2))],-1);delta=(self.fusion_kv(ctx).transpose(1,2)+torch.tanh(self.direct_scale_kv)[None,:,None]*diff)*torch.sigmoid(self.gate_kv(ctx)).transpose(1,2)*node_support[:,None]
        if disable_memory:delta=delta*0
        pred=base+self.norm_std*delta;memory_only=periodic+self.norm_std*mem
        return pred,{'base':base,'periodic':periodic,'memory_only':memory_only,'delta':delta,'support':node_support,'edge_attention':edge_att,'episode_weights':node_w,'retrieval_ids':ids,'retrieval_alpha':alpha}

def set_scope(model,warmup=False):
    for n,p in model.named_parameters():
        backbone=n in ('daily','weekly') or n.startswith(('temporal_encoder.','temporal_head.'));p.requires_grad_(not (warmup and backbone))
        if model.mode=='global' and n.startswith(('edge_family','edge_attention_kv','edge_encoder_kv','family_embedding','size_embedding')):p.requires_grad_(False)
        if model.mode=='physical' and n=='global_metric_kv':p.requires_grad_(False)
