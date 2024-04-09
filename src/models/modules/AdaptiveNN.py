import torch as th
from torch import nn

import dgl
import dgl.function as fn

class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats) -> None:
        super(MLP, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self._build_layers()
        
    def _build_layers(self):
        self.fc1 = nn.Linear(self.in_feats, self.hid_feats)
        self.bn1 = nn.BatchNorm1d(self.hid_feats)
        self.dropout1 = nn.Dropout(p=0.2)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(self.hid_feats, self.out_feats)
        self.bn2 = nn.BatchNorm1d(self.out_feats)
        self.elu2 = nn.ELU()
        
    def forward(self, x):
        out = self.fc1(x)
        
        #out = self.bn1(out) if out.size(0) > 1 else out
        out = self.elu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        #out = self.bn2(out) if out.size(0) > 1 else out
        out = self.elu2(out)
        return out

class AdaptiveNN(nn.Module):
    def __init__(self, api_feat_size, mashup_feat_size, hid_feats=256) -> None:
        super(AdaptiveNN, self).__init__()
        self.api_feat_size = api_feat_size
        self.mashup_feat_size = mashup_feat_size
        self.hid_feats = hid_feats
        self._build_layers()
    
    def _build_layers(self):
        self.mlp1 = MLP(self.api_feat_size + self.mashup_feat_size, self.hid_feats, self.api_feat_size) # [h1; m] -> [h]
        self.mlp2 = MLP(self.api_feat_size * 2, self.hid_feats, self.api_feat_size)
        self.mlp3 = MLP(self.api_feat_size, self.hid_feats, self.api_feat_size)
        self.mlp4 = MLP(self.api_feat_size*3, self.hid_feats, self.api_feat_size)
        self.fc = nn.Linear(self.api_feat_size, 1)
        
    def forward(self, graph: dgl.DGLGraph, node_feats: th.Tensor, graph_feats: th.Tensor):
        graph = graph.local_var()
        graph_feats = graph_feats.view(1, -1).repeat(node_feats.size(0), 1)
        in_feats = th.cat([node_feats, graph_feats], dim=1)
        out = self.mlp1(in_feats) # The first step
        graph.ndata['feats'] = out
        graph.apply_edges(lambda edges: {"ef": th.cat([edges.src['feats'], edges.dst['feats']],dim=-1)})         # node to edge
        graph.edata['skip_e'] = self.mlp2(graph.edata["ef"])
        graph.update_all(fn.copy_e('skip_e', 'm'), fn.sum('m', 'feats')) # edge to node
        graph.ndata['feats'] = self.mlp3(graph.ndata['feats']) # mlp3(x)
        graph.apply_edges(lambda edges: {"ef": th.cat([edges.src['feats'], edges.dst['feats']],dim=-1)})
        graph.edata["ef"] = self.mlp4(th.cat((graph.edata['ef'], graph.edata['skip_e']), dim=-1)) # mlp4(x) with skip connect # type: ignore
        # graph.edata["ef"] = self.mlp4((graph.edata['ef']))
        edge_weights = nn.Sigmoid()(self.fc(graph.edata['ef']))
        return edge_weights, graph.ndata['feats']
        
        