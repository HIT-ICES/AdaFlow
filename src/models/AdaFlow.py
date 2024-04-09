import os

import pytorch_lightning as pl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import pandas as pd
import wandb
import json


import dgl
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import PNAConv, GATConv

from src.utils.metrics import Precision, NormalizedDCG, Recall, MILC

from src.models.modules.AdaptiveNN import AdaptiveNN, MLP
from src.models.modules.TransformerConv import TransformerConv

class AdaFlow(pl.LightningModule):
    
    def __init__(self,
                 num_api,
                 num_mashups,
                 feat_size,
                 lr,
                 weight_decay,
                 frozen_api_embeds=False,
                 edge_rm_threhold=0.5,
                 update_after_test=False,
                 top_k=5,
                 save_path = None,
                 metric_path=None,
                 seed = None,
                 ) -> None:
        super(AdaFlow, self).__init__()
        self.save_hyperparameters()
        self.num_api = num_api
        self.num_mashups = num_mashups
        self.feat_size = feat_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_rm_threhold = edge_rm_threhold
        self.update_after_test = update_after_test
        self.heapmap_data = None
        self.top_k = top_k
        self.save_path = save_path
        self.metric_path = metric_path
        self.seed = seed
        
        # [M, D]
        self.register_buffer("mashup_embeds", th.from_numpy(
            np.load("data/PW/mashup_dev_embeds.npz", allow_pickle=True)['arr_0']
        ).float())
        
        # [A, D]
        if frozen_api_embeds:
            self.register_buffer("api_embeds", th.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float())
        else:
            self.register_parameter("api_embeds", nn.Parameter(th.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float()))

        self.P5_test = Precision(top_k=5).to(self.device)
        self.R5_test = Recall(top_k=5).to(self.device)
        self.nDCG5_test = NormalizedDCG(top_k=5).to(self.device)
        self.MILC5_test = MILC(top_k=5).to(self.device)
        self.P5_val_0 = Precision(top_k=5).to(self.device)
        self.R5_val_0 = Recall(top_k=5).to(self.device)
        self.nDCG5_val_0 = NormalizedDCG(top_k=5).to(self.device)
        self.MILC5_val_0 = MILC(top_k=5).to(self.device)
        self.P5_val_1 = Precision(top_k=5).to(self.device)
        self.R5_val_1 = Recall(top_k=5).to(self.device)
        self.nDCG5_val_1 = NormalizedDCG(top_k=5).to(self.device)
        self.MILC5_val_1 = MILC(top_k=5).to(self.device)
        self.train_escape = 0
        self.test_escape = 0

        self.mashup_feat_size = self.mashup_embeds.size(1) # type: ignore[misc]
        self.api_feat_size = self.api_embeds.size(1) # type: ignore[misc]
        
        self.register_buffer("api_embeds_final", th.zeros(self.num_api, self.feat_size))
        
        self._build_layers()

        self.idx_0 = 0
        self.idx_1 = 0
        self.epoch = 0

    def _build_layers(self):
        self.m_fc = nn.Linear(self.mashup_feat_size, self.feat_size) # [M, D] -> [M, F]
        self.a_fc = nn.Linear(self.api_feat_size, self.feat_size) # [A, D] -> [A, F]
        self.adaptive_nn = AdaptiveNN(api_feat_size=self.feat_size, mashup_feat_size=self.feat_size)
        self.rnncell = nn.GRUCell(self.feat_size, self.feat_size)
        self.infer_mlp = MLP(in_feats=self.feat_size, hid_feats=256, out_feats=self.num_api)
        self.infer_mlp_pri = MLP(in_feats=5*self.feat_size, hid_feats=256, out_feats=self.num_api)
        self.gnn_conv = TransformerConv(in_feats=self.feat_size, out_feats=self.feat_size, num_heads=1, edge_feats=1)

    def on_train_start(self):
        self.propensity_score = self.trainer.datamodule.propensity_score
        self.propensity_score = self.propensity_score.to(self.device)

        self.PSR5_test = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_test = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_test = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_0 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_0 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_0 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_1 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_1 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_1 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)

    def on_train_epoch_start(self):
        self.api_embeds_final = self.api_embeds_final * 0 # type: ignore[misc]

    def training_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        
        mashup_feats = self.mashup_embeds[graph_labels] # type: ignore[misc]
        start_time = time.time()
        mashup_feats = self.m_fc(mashup_feats) # reduce dim
        api_feats = self.a_fc(self.api_embeds) # reduce dim
        self.train_escape += time.time() - start_time
        
        batch_graph.ndata['feats'] = api_feats[batch_graph.ndata[dgl.NID]] # type: ignore[misc]
        graphs = dgl.unbatch(batch_graph)
        
        infered_graphs = [] # graphs after remove edges
        loss = 0
        start_time = time.time()
        for i, (graph, mashup) in enumerate(zip(graphs, mashup_feats)):
            node_feats = graph.ndata['feats']
            # TODO(@mingyi): do we need to update the node embeddings?
            edge_weights, node_feats = self.adaptive_nn(graph, node_feats, mashup)
            graph.edata['weight'] = edge_weights
            graph = self._remove_edges_from_graph(graph)
            infered_graphs.append(graph)
        graph_merged = self._merge_graphs(infered_graphs)
        node_feats = self.gnn_conv(graph_merged, api_feats, graph_merged.edata['weight'])
        node_feats = th.squeeze(node_feats)
        graph_merged.ndata['feats'] = node_feats #
        hiddens = self.rnncell(node_feats, self.api_embeds_final)
        self.api_embeds_final = hiddens.detach().clone()
        
        for graph, mashup in zip(graphs, mashup_feats):
            graph.ndata['hx'] = hiddens[graph.ndata[dgl.NID]]
            loss += self.loss_compution(graph, mashup)
        self.train_escape += time.time() - start_time
        loss /= len(graphs)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)
        return loss
            
    def loss_compution(self, graph, mashup):
        loss = 0
        api_freq = graph.ndata['freq']
        api_index = th.argsort(api_freq)
        x_s, hx_s = graph.ndata['feats'][api_index], graph.ndata['hx'][api_index]
        api_idxs = graph.ndata[dgl.NID][api_index]
        session_apis, last_api = [], None
        for x, hx, api_idx in zip(x_s, hx_s, api_idxs):
            x_star = th.concat((x, hx), dim=-1)
            if last_api is None:
                predict = self.infer_mlp(mashup.view(1, -1))
                predict_prob = F.softmax(predict, dim=1).view(-1)
                loss -= th.log(predict_prob[api_idx] + 1e-10)
            else:
                session_pooled, _ = th.max(th.stack(session_apis, dim=0), dim=0)
                session_pooled = session_pooled.view(1, -1)
                in_feats = th.cat([mashup.view(1, -1), session_pooled, last_api], dim=1)
                predict = self.infer_mlp_pri(in_feats)
                predict_prob = F.softmax(predict, dim=1).view(-1)
                loss -= th.log(predict_prob[api_idx] + 1e-10)
            last_api = x_star.view(1, -1)
            session_apis.append(x_star)
        return loss

    def _remove_edges_from_graph(self, graph):
        src, dst = graph.edges()
        weights = graph.edata['weight']
        weight_sum = 0
        num = 0
        weight_average = 0
        for weight, x, y in zip(weights, src, dst):
            if x.item() == y.item():
                continue
            weight_sum += weight.item()
            num += 1
        if num != 0:
            weight_average = weight_sum / num
        edge_masks = (graph.edata['weight'] < weight_average).view(-1)
        edge_masks = edge_masks.nonzero().squeeze()
        graph = dgl.transforms.remove_edges(graph, edge_masks)
        return graph
    
    def _merge_graphs(self, graphs):
        graph_all = dgl.graph(([], []), num_nodes=self.num_api)
        node_ids = th.arange(self.num_api)
        src, dst = [], []
        for graph in graphs:
            edges = graph.edges()
            src.append(edges[0])
            dst.append(edges[1])
        src = th.cat(src, 0).cpu()
        dst = th.cat(dst, 0).cpu()
        graph_all.add_edges(src, dst)
        graph_all.edata['weight'] = th.zeros(graph_all.num_edges(), 1).float()

        for graph in graphs:
            edges = graph.edges()
            eids = graph_all.edge_ids(edges[0].cpu(), edges[1].cpu())
            weights = graph.edata['weight'].cpu()
            g_all_w = graph_all.edata['weight']
            g_all_w[eids] = g_all_w[eids] + weights
        
        graph_all = graph_all.to(self.device)
        graph = dgl.remove_self_loop(graph_all)
        graph = dgl.add_self_loop(graph, edge_feat_names=['weight'])
        return graph
    def validation_step(self, batch, batch_idx, loader_idx):
        batch_graph, graph_labels = batch
        
        batch_graph.ndata['feats'] = self.api_embeds[batch_graph.ndata[dgl.NID]] # type: ignore[misc]
        batch_graph.ndata['feats'] = self.a_fc(batch_graph.ndata['feats'])
        
        mashup_feats = self.mashup_embeds[graph_labels] # type: ignore[misc]
        mashup_feats = self.m_fc(mashup_feats)
        
        api_x = self.a_fc(self.api_embeds)
        api_hx = self.api_embeds_final  # hidden
        api_star = th.cat([api_x, api_hx], dim=1)
        graphs = dgl.unbatch(batch_graph)
        predict_list, logist = [], []
        for graph, mashup in zip(graphs, mashup_feats):
            pred_onehot = th.zeros(self.num_api).float()
            logist_onehot = th.zeros(self.num_api).float()
            logist_onehot[graph.ndata[dgl.NID].cpu()] = 1.
            recomended_apis, session_apis, last_api = [], [], None
            mashup_exp = mashup.view(1, -1)
            predict = self.infer_mlp(mashup_exp)
            
            predict_prob = F.softmax(predict, dim=1).view(-1) # [num_api]
            rec_api_idx = th.argmax(predict_prob)
            pred_onehot[rec_api_idx.cpu()] = 1. 
            
            mask_invked = th.zeros(self.num_api).type_as(mashup).float()
            mask_invked[rec_api_idx] = float('inf')
            
            recomended_apis.append(rec_api_idx)
            session_apis.append(api_star[rec_api_idx])
            last_api = api_star[rec_api_idx].view(1, -1)
            
            for i in range(4):
                session_pooled, _ = th.max(th.stack(session_apis, dim=0), dim=0)
                session_pooled = session_pooled.view(1, -1)
                in_feats = th.cat([mashup_exp, session_pooled, last_api], dim=1)
                predict = self.infer_mlp_pri(in_feats)
                predict = predict - mask_invked.view(1, -1)
                predict_prob = F.softmax(predict, dim=1).view(-1)
                rec_api_idx = th.argmax(predict_prob)
                pred_onehot[rec_api_idx.cpu()] = 1.
                mask_invked[rec_api_idx] = float('inf')
                recomended_apis.append(rec_api_idx.cpu())
                session_apis.append(api_star[rec_api_idx])
                last_api = api_star[rec_api_idx].view(1, -1)
                
            predict_list.append(pred_onehot)
            logist.append(logist_onehot)

        p_t = th.stack(predict_list, dim=0).to(self.device)
        l_t = th.stack(logist, dim=0).to(self.device)
        if not self.trainer.sanity_checking:
            if loader_idx == 0:
                self.R5_val_0.update(p_t, l_t)
                self.P5_val_0.update(p_t, l_t)
                self.PSP5_val_0.update(p_t, l_t)
                self.log("val/P@5", self.P5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_0, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_0, on_step=False, on_epoch=True)
            else:
                self.R5_val_1.update(p_t, l_t)
                self.P5_val_1.update(p_t, l_t)
                self.PSP5_val_1.update(p_t, l_t)
                self.log("val/P@5", self.P5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_1, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_1, on_step=False, on_epoch=True)
        if self.update_after_test:
            self._update_after_test(batch)

    def test_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        batch_graph.ndata['feats'] = self.api_embeds[batch_graph.ndata[dgl.NID]] # type: ignore[misc]
        start_time = time.time()
        batch_graph.ndata['feats'] = self.a_fc(batch_graph.ndata['feats'])
        mashup_feats = self.mashup_embeds[graph_labels] # type: ignore[misc]
        mashup_feats = self.m_fc(mashup_feats)
        api_x = self.a_fc(self.api_embeds)
        api_hx = self.api_embeds_final
        api_star = th.cat([api_x, api_hx], dim=1)
        
        graphs = dgl.unbatch(batch_graph)
        predict_list, logist = [], []
        for graph, mashup, graph_label in zip(graphs, mashup_feats, graph_labels):
            pred_onehot = th.zeros(self.num_api).float()
            logist_onehot = th.zeros(self.num_api).float()
            logist_onehot[graph.ndata[dgl.NID].cpu()] = 1.
            recomended_apis, session_apis, last_api = [], [], None
            mashup_exp = mashup.view(1, -1)
            predict = self.infer_mlp(mashup_exp)
            predict_prob = F.softmax(predict, dim=1).view(-1) # [num_api]
            rec_api_idx = th.argmax(predict_prob)
            pred_onehot[rec_api_idx.cpu()] = 1.
            
            mask_invked = th.zeros(self.num_api).type_as(mashup).float()
            mask_invked[rec_api_idx] = float('inf')
            recomended_apis.append(rec_api_idx)
            session_apis.append(api_star[rec_api_idx])
            last_api = api_star[rec_api_idx].view(1, -1)
            
            for _ in range(4):
                session_pooled, _ = th.max(th.stack(session_apis, dim=0), dim=0)
                session_pooled = session_pooled.view(1, -1)
                in_feats = th.cat([mashup_exp, session_pooled, last_api], dim=1)
                predict = self.infer_mlp_pri(in_feats)
                predict = predict - mask_invked.view(1, -1)
                predict_prob = F.softmax(predict, dim=1).view(-1)
                rec_api_idx = th.argmax(predict_prob)
                mask_invked[rec_api_idx] = float('inf')
                pred_onehot[rec_api_idx.cpu()] = 1. 
                recomended_apis.append(rec_api_idx.cpu())
                session_apis.append(api_star[rec_api_idx])
                last_api = api_star[rec_api_idx].view(1, -1)
                
            predict_list.append(pred_onehot)
            logist.append(logist_onehot)

        self.test_escape += time.time() - start_time
        p_t = th.stack(predict_list, dim=0).to(self.device)
        l_t = th.stack(logist, dim=0).to(self.device)
        self.P5_test.update(p_t, l_t)
        self.PSP5_test.update(p_t, l_t)
        self.R5_test.update(p_t, l_t)
        self.log("test/P@5", self.P5_test, on_step=False, on_epoch=True)
        self.log("test/PSP@5", self.PSP5_test, on_step=False, on_epoch=True)
        self.log("test/R@5", self.R5_test, on_step=False, on_epoch=True)
        if self.update_after_test:
            self._update_after_test(batch)

    def _update_after_test(self, batch):
        batch_graph, graph_labels = batch
        
        mashup_feats = self.mashup_embeds[graph_labels] # type: ignore[misc]
        mashup_feats = self.m_fc(mashup_feats)
        
        api_feats = self.a_fc(self.api_embeds)
        
        batch_graph.ndata['feats'] = api_feats[batch_graph.ndata[dgl.NID]] # type: ignore[misc]
        graphs = dgl.unbatch(batch_graph)
        
        infered_graphs = []
        loss = 0
        for graph, mashup in zip(graphs, mashup_feats):
            node_feats = graph.ndata['feats']
            edge_weights, node_feats = self.adaptive_nn(graph, node_feats, mashup)
            graph.edata['weight'] = edge_weights
            graph = self._remove_edges_from_graph(graph)
            graph = dgl.add_self_loop(graph, ['weight'], fill_data=1)
            infered_graphs.append(graph)
        graph_merged = self._merge_graphs(infered_graphs)
        node_feats = self.gnn_conv(graph_merged, api_feats, graph_merged.edata['weight'])
        node_feats = th.squeeze(node_feats)
        graph_merged.ndata['feats'] = node_feats
        hiddens = self.rnncell(node_feats, self.api_embeds_final)    
        self.api_embeds_final = hiddens.detach().clone()
    
    def configure_optimizers(self):
        return th.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )