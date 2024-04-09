from select import select
from turtle import update
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import time
import os
import csv
import json

from src.utils.metrics import Precision, NormalizedDCG, Recall, MILC
from torch_geometric.nn import TransformerConv, Sequential, LayerNorm
from torch_geometric.utils import dropout_adj
from torch_geometric.nn.inits import glorot, zeros
import dgl


class DySR(pl.LightningModule):
    def __init__(self,
                 negative_sample=5,
                 api_out_channels=128,
                 mashup_out_channels=128,
                 heads=1,
                 edge_dropout=0.2,
                 dropout_test=0.0,
                 lr=1e-3,
                 api_graph=True,  # this should be set to true.
                 penate_align=False,
                 constrain_agg='max',  # max would work better.
                 constrain_heads=None,
                 api_graph_agg='mean',
                 top_k=5,  # how many top apis to recomender
                 gnn_layers=1,
                 order='descending',
                 semantic_path: str = None,
                 update_test=True,
                 edge_message=True,
                 edge_message_agg='mean',
                 dynamic_api=True,
                 weight_decay=1e-4,
                 hidden_channels = 512,
                 fix_api_embed = True,
                 metric_path = None,
                 seed = None,
                 save_path = None,
                 ) -> None:
        super().__init__()
        self.save_path = save_path
        self.save_hyperparameters()
        self.register_buffer('mashup_embeds',
                             torch.from_numpy(np.load("data/PW/mashup_dev_embeds.npz", allow_pickle=True)['arr_0']).float())
        if fix_api_embed:
            self.register_buffer('api_embeds', torch.from_numpy(np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']).float())
        else:
            self.register_parameter('api_embeds', Parameter(torch.from_numpy(np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']).float()))

        self.mashup_embed_channels = self.mashup_embeds.size(1)
        self.num_api = self.api_embeds.size(0)
        self.api_out_channels = api_out_channels
        self.mashup_out_channels = mashup_out_channels
        self.heads = heads
        self.edge_dropout = edge_dropout
        self.gnn_layers = gnn_layers

        self.dynamic_api = dynamic_api

        self.dropout_test = dropout_test
        self.update_test = update_test
        self.order = order

        self.constrain_heads = constrain_heads

        self.propensity_score = 1
        self.negative_sample = negative_sample
        self.lr = lr
        self.weight_decay = weight_decay
        self.top_k = top_k
        self.api_graph = api_graph
        self.penate_align = penate_align
        self.constrain_agg = constrain_agg
        self.api_graph_agg = api_graph_agg
        self.semantic_path = semantic_path
        self.edge_message = edge_message
        self.edge_message_agg = edge_message_agg
        self.hidden_channels = hidden_channels
        self.metric_path = metric_path
        self.seed = seed

        self.register_buffer('hidden', torch.zeros(self.num_api, self.api_out_channels))
        zeros(self.hidden)

        # used to mask un-invoked APIs
        self.register_buffer('api_mask', torch.zeros(self.num_api).long())

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


        self._build_layers()

        self.idx_0 = 0
        self.idx_1 = 0
        self.epoch = 0

    def _build_layers(self):
        # align embeds
        self.align_mlp = nn.Sequential(
            nn.Linear(self.mashup_embed_channels, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.mashup_out_channels)
        )

        self.api_reduce_dem_mlp = nn.Sequential(
            nn.Linear(self.api_embeds.size(1), self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.api_out_channels)
        )

        gnn_layers = []
        for i in range(self.gnn_layers):
            gnn_layers.append(TransformerConv(self.api_out_channels if i == 0 else self.api_out_channels,
                                              int(self.api_out_channels / self.heads), self.heads,
                                              edge_dim=self.api_out_channels if self.edge_message else None))
            gnn_layers.append(LayerNorm(self.api_out_channels))
        self.gnns = nn.ModuleList(gnn_layers)

        """
        self.gnn_1 = TransformerConv(self.api_embed_channels, int(self.api_out_channels/self.heads), self.heads, edge_dim=self.api_out_channels)
        self.layerNorm_1 = LayerNorm(self.api_out_channels)
        self.gnn_2 = TransformerConv(self.api_out_channels, int(self.api_out_channels/self.heads), self.heads, edge_dim=self.api_out_channels)
        self.layerNorm_2 = LayerNorm(self.api_out_channels)
        """
        self.rnn = nn.RNNCell(self.api_out_channels, self.api_out_channels)

        # p(y|G_{<t}, x)
        self.infer_linear = nn.Sequential(
            nn.Linear(self.api_out_channels * 2 + self.mashup_out_channels if self.api_graph else self.api_out_channels,
                      self.api_out_channels * 3),
            nn.ReLU(),
            nn.Linear(self.api_out_channels * 3, self.num_api)
        )
        self.infer_linear_pri = nn.Sequential(
            nn.Linear(
                self.api_out_channels * 3 + (self.api_out_channels + self.mashup_out_channels) if self.api_graph else self.api_out_channels * 2 + self.api_out_channels,
                self.api_out_channels * 3),
            nn.ReLU(),
            nn.Linear(self.api_out_channels * 3, self.num_api)
        )
        if self.constrain_heads is not None:
            self.attention_constrain = nn.MultiheadAttention(self.api_out_channels + self.api_out_channels,
                                                             self.constrain_heads)

    def on_train_start(self) -> None:
        self.propensity_score = self.trainer.datamodule.propensity_score
        self.propensity_score = self.propensity_score.to(self.device)

        self.PSP5_test = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_test = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_test = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_0 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_0 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_0 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_1 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_1 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_1 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.idx_0 = self.P5_val_0.compute()
            self.idx_1 = self.P5_val_1.compute()
            self.metric_csv.writerow([
                self.epoch, self.idx_0.cpu().item(), self.idx_1.cpu().item()
            ])
            self.epoch += 1


    def on_train_epoch_start(self) -> None:
        self.train_escape = 0
        zeros(self.hidden)
        zeros(self.api_mask)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)
        Ys = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.zeros(self.num_api).type_as(self.mashup_embeds).float()
            masks[pos_nodes] = 1  # type: ignore
            Ys.append(masks)
        Ys = torch.stack(Ys, dim=0)
        Xs = self.mashup_embeds[graph_labels]
        start_time = time.time()
        Xs = self.align_mlp(Xs)
        api_embeds = self.api_reduce_dem_mlp(self.api_embeds) if self.api_out_channels != 300 else self.api_embeds
        self.train_escape += time.time() - start_time
        edge_index, edge_attr = self.format_samples(Xs, Ys)
        start_time = time.time()
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.edge_dropout, force_undirected=True)

        # api_embeds_t = api_embeds
        edge_index = edge_index.to(self.device)
        for i in range(self.gnn_layers):
            api_embeds = self.gnns[i * 2](api_embeds, edge_index, edge_attr if self.edge_message else None)
            api_embeds = self.gnns[i * 2 + 1](api_embeds)

        hidden = self.rnn(api_embeds, self.hidden)

        self.hidden = hidden.detach()
        s_bar = torch.cat([hidden, api_embeds], dim=1)
        # ==========================================================
        # 消融实验-api_social_enviroment: 考虑不同的 g 生成方法带来的影响
        # ==========================================================
        g = self.generate_api_graph_embedds(s_bar)

        loss = 0
        sample_num = 0
        for x, y in zip(Xs, Ys):
            sample_num += 1
            prob = []
            # =====================================================
            # 消融实验-api_social_enviroment: 考虑是否使用 g 带来的影响
            # =====================================================
            in_feature = self.generate_background_features(x, g)

            y = y * self.propensity_score
            invoked_apis = y.nonzero(as_tuple=True)[0]
            k = len(invoked_apis)
            top_k, top_k_idx = torch.topk(y, k=k, dim=0, sorted=True)
            # =====================================================
            # 消融实验-API选取顺序: 考虑不同选择策略对推荐产生的影响
            # =====================================================
            top_k_idx = self.api_select_order(k, top_k_idx)


            if self.constrain_heads is None:
                api_constraint = s_bar[top_k_idx[0]]
            mask_invked = torch.zeros(self.num_api).type_as(x).float()
            selected_apis_embedds = []
            for index, api_idx in enumerate(top_k_idx):
                if index == 0:
                    first_api_prob = self.infer_linear(in_feature)
                    first_api_prob = F.softmax(first_api_prob, dim=1).view(-1)
                    p = first_api_prob[api_idx]
                    loss = loss + torch.log(p)
                    mask_invked[api_idx] = float('inf')
                    selected_apis_embedds.append(s_bar[api_idx])
                else:
                    if self.constrain_heads is not None:
                        key = torch.stack(selected_apis_embedds)
                        key = key.unsqueeze(1)
                        attn_out, _ = self.attention_constrain(key, key, key)
                        attn_out = torch.mean(attn_out, dim=0)
                        in_feature_constrain = torch.cat([in_feature, attn_out.view(1, -1)], dim=1)
                    elif self.constrain_agg == 'mean':
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1) / index], dim=1)
                    elif self.constrain_agg:
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1)], dim=1)
                    # mask invoked apis
                    next_api_prob = self.infer_linear_pri(in_feature_constrain)
                    next_api_prob = next_api_prob - mask_invked.view(1, -1)
                    next_api_prob = F.softmax(next_api_prob, dim=1).view(-1)
                    mask_invked[api_idx] = float('inf')
                    selected_apis_embedds.append(s_bar[api_idx])
                    p = next_api_prob[api_idx]
                    loss = loss + torch.log(p)
                    if self.constrain_heads is not None:
                        pass
                    elif self.constrain_agg == 'mean':
                        api_constraint = api_constraint + s_bar[api_idx]
                    elif self.constrain_agg == 'max':
                        api_constraint = torch.max(api_constraint, s_bar[api_idx])
                    elif self.constrain_agg == 'sum':
                        api_constraint = api_constraint + s_bar[api_idx]

        self.api_mask[edge_index[0]] = 1
        loss = -loss / sample_num
        self.train_escape += time.time() - start_time
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)
        Ys = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.zeros(self.num_api).type_as(self.mashup_embeds).float()
            masks[pos_nodes] = 1 # type: ignore
            Ys.append(masks)
        Ys = torch.stack(Ys, dim=0)
        Xs = self.mashup_embeds[graph_labels]
        Xs = self.align_mlp(Xs)
        api_embeds = self.api_reduce_dem_mlp(self.api_embeds) if self.api_out_channels != 300 else self.api_embeds
        edge_index, edge_attr = self.format_samples(Xs, Ys)
        s_bar = torch.cat([self.hidden, api_embeds], dim=1)

        g = self.generate_api_graph_embedds(s_bar)

        preds = []
        for x, y in zip(Xs, Ys):
            in_feature = self.generate_background_features(x, g)

            recommend, values = [], []
            mask_invked = torch.zeros(self.num_api).type_as(x).float()
            selected_apis_embedds = []
            for i in range(self.top_k):
                if i == 0:
                    first_api_prob = F.softmax(self.infer_linear(in_feature), dim=1).view(-1)
                    max_id = torch.argmax(first_api_prob)
                    api_constraint = s_bar[max_id]
                    mask_invked[max_id] = float('inf')
                    recommend.append(max_id)
                    values.append(np.exp(-i))
                    selected_apis_embedds.append(s_bar[max_id])
                else:
                    if self.constrain_heads is not None:
                        key = torch.stack(selected_apis_embedds)
                        key = key.unsqueeze(1)
                        attn_out, _ = self.attention_constrain(key, key, key)
                        attn_out = torch.mean(attn_out, dim=0)
                        in_feature_constrain = torch.cat([in_feature, attn_out.view(1, -1)], dim=1)
                    elif self.constrain_agg == 'mean':
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1) / i], dim=1)
                    elif self.constrain_agg:
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1)], dim=1)
                    next_api_prob = self.infer_linear_pri(in_feature_constrain)
                    next_api_prob = next_api_prob - mask_invked.view(1, -1)
                    next_api_prob = F.softmax(next_api_prob, dim=1).view(-1)
                    max_id = torch.argmax(next_api_prob)
                    mask_invked[max_id] = float('inf')
                    selected_apis_embedds.append(s_bar[max_id])
                    if self.constrain_agg == 'mean':
                        api_constraint = api_constraint + s_bar[max_id]
                    elif self.constrain_agg == 'max':
                        api_constraint = torch.max(api_constraint, s_bar[max_id])
                    elif self.constrain_agg == 'sum':
                        api_constraint = api_constraint + s_bar[max_id]
                    recommend.append(max_id)
                    values.append(np.exp(-i))
            preds.append(torch.zeros(self.num_api).float().scatter(0, torch.tensor(recommend),
                                                                   torch.tensor(values).float()).view(1, -1))
        preds = torch.cat(preds, dim=0).type_as(Xs).float()
        if not self.trainer.sanity_checking:
            if loader_idx == 0:
                self.P5_val_0.update(preds, Ys)
                self.PSP5_val_0.update(preds, Ys)
                self.R5_val_0.update(preds, Ys)
                self.log("val/P@5", self.P5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_0, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_0, on_step=False, on_epoch=True)
            else:
                self.P5_val_1.update(preds, Ys)
                self.PSP5_val_1.update(preds, Ys)
                self.R5_val_1.update(preds, Ys)
                self.log("val/P@5", self.P5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_1, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_1, on_step=False, on_epoch=True)
        if self.update_test:
            self.update_hidden(edge_index, edge_attr, dropout=self.dropout_test)

    def on_test_start(self) -> None:
        self.test_escap = 0
        self.mashup_real_preds = []

    def test_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)

        Ys = []
        pos_nodes_list = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            pos_nodes_list.append(pos_nodes.cpu().tolist())
            masks = torch.zeros(self.num_api).type_as(self.mashup_embeds).float()
            masks[pos_nodes] = 1  # type: ignore
            Ys.append(masks)
        Ys = torch.stack(Ys, dim=0)
        Xs = self.mashup_embeds[graph_labels]
        start_time = time.time()
        Xs = self.align_mlp(Xs)
        api_embeds = self.api_reduce_dem_mlp(self.api_embeds) if self.api_out_channels != 300 else self.api_embeds
        self.test_escape += time.time() - start_time
        edge_index, edge_attr = self.format_samples(Xs, Ys)
        start_time = time.time()
        s_bar = torch.cat([self.hidden, api_embeds], dim=1)  # [N, 2C']

        g = self.generate_api_graph_embedds(s_bar)

        preds = []
        for x, y, graph_label, pos_nodes in zip(Xs, Ys, graph_labels, pos_nodes_list):
            in_feature = self.generate_background_features(x, g)
            recommend, values = [], []
            mask_invked = torch.zeros(self.num_api).type_as(x).float()
            selected_apis_embedds = []
            for i in range(self.top_k):
                if i == 0:
                    first_api_prob = F.softmax(self.infer_linear(in_feature), dim=1).view(-1)
                    max_id = torch.argmax(first_api_prob)
                    pred.append(max_id.cpu().tolist())
                    api_constraint = s_bar[max_id]
                    recommend.append(max_id)
                    values.append(np.exp(-i))
                    selected_apis_embedds.append(s_bar[max_id])
                else:
                    if self.constrain_heads is not None:
                        key = torch.stack(selected_apis_embedds)
                        key = key.unsqueeze(1)
                        attn_out, _ = self.attention_constrain(key, key, key)
                        attn_out = torch.mean(attn_out, dim=0)
                        in_feature_constrain = torch.cat([in_feature, attn_out.view(1, -1)], dim=1)
                    elif self.constrain_agg == 'mean':
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1) / i], dim=1)
                    elif self.constrain_agg:
                        in_feature_constrain = torch.cat([in_feature, api_constraint.view(1, -1)], dim=1)
                    # ====================================
                    # mask invoked apis
                    # ====================================
                    next_api_prob = self.infer_linear_pri(in_feature_constrain)
                    next_api_prob = next_api_prob - mask_invked.view(1, -1)
                    next_api_prob = F.softmax(next_api_prob, dim=1).view(-1)
                    max_id = torch.argmax(next_api_prob)
                    pred.append(max_id.cpu().tolist())
                    mask_invked[max_id] = float('inf')
                    selected_apis_embedds.append(s_bar[max_id])
                    if self.constrain_agg == 'mean':
                        api_constraint = api_constraint + s_bar[max_id]
                    elif self.constrain_agg == 'max':
                        api_constraint = torch.max(api_constraint, s_bar[max_id])
                    elif self.constrain_agg == 'sum':
                        api_constraint = api_constraint + s_bar[max_id]
                    recommend.append(max_id)
                    values.append(np.exp(-i))
            preds.append(torch.zeros(self.num_api).float().scatter(0, torch.tensor(recommend),
                                                                   torch.tensor(values).float()).view(1, -1))
        preds = torch.cat(preds, dim=0).type_as(Xs).float()
        # ==================
        self.test_escap += time.time() - start_time
        self.P5_test.update(preds, Ys)
        self.PSP5_test.update(preds, Ys)
        self.R5_test.update(preds, Ys)
        self.log("test/P@5", self.P5_test, on_step=False, on_epoch=True)
        self.log("test/PSP@5", self.PSP5_test, on_step=False, on_epoch=True)
        self.log("test/R@5", self.R5_test, on_step=False, on_epoch=True)
        if self.update_test:
            self.update_hidden(edge_index, edge_attr, dropout=self.dropout_test)

    def update_hidden(self, edge_index, edge_attr, dropout):
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=dropout, force_undirected=True)
        edge_index = edge_index.to(self.device)
        api_embeds = self.api_reduce_dem_mlp(self.api_embeds) if self.api_out_channels != 300 else self.api_embeds
        for i in range(self.gnn_layers):
            api_embeds = self.gnns[i * 2](api_embeds, edge_index, edge_attr if self.edge_message else None)
            api_embeds = self.gnns[i * 2 + 1](api_embeds)
        self.hidden = self.rnn(api_embeds, self.hidden)
        self.api_mask[edge_index[0]] = 1

    def generate_api_graph_embedds(self, s_bar):
        if torch.sum(self.api_mask) == 0:
            g = torch.zeros(s_bar.size(1)).type_as(self.mashup_embeds)
        else:
            if self.api_graph_agg == 'mean':
                g = torch.mean(s_bar[self.api_mask == 1], dim=0)
            elif self.api_graph_agg == 'max':
                g = torch.max(s_bar[self.api_mask == 1], dim=0)[0]
        return g

    def generate_background_features(self, x, g):
        if self.api_graph:
            in_feature = torch.cat([x, g], dim=0).view(1, -1)
        else:
            in_feature = x.view(1, -1)
        return in_feature

    def api_select_order(self, k, top_k_idx):
        if self.order == 'descending':
            top_k_idx = top_k_idx
        elif self.order == 'ascending':
            top_k_idx = top_k_idx.flip(0)
        elif self.order == 'random':
            shuffle_idx = torch.randperm(k)
            top_k_idx = top_k_idx[shuffle_idx]
        return top_k_idx

    def format_samples(self, Xs, Ys, **kwargs):
        """
        Xs: [B, C]  # C: message channel
        Ys: [B, N]  # N: API numbers
        """
        edge_message = {}
        for x, y in zip(Xs, Ys):
            invoked_apis = y.nonzero(as_tuple=True)[0]
            invoked_apis = invoked_apis.cpu().numpy()
            for api_i in invoked_apis:
                for api_j in invoked_apis:
                    edge_message[(api_i, api_j)] = edge_message.get((api_i, api_j), []) + [x]
        src, dst, edge_attr = [], [], []
        for (u, v), messages in edge_message.items():
            src.append(u)
            dst.append(v)
            if self.edge_message_agg == 'mean':
                edge_attr.append(torch.mean(torch.stack(messages, dim=0), dim=0))
            else:
                edge_attr.append(torch.sum(torch.stack(messages, dim=0), dim=0))
        src = torch.tensor(src)
        dsc = torch.tensor(dst)
        edge_index = torch.stack([src, dsc], dim=0)
        edge_attr = torch.stack(edge_attr, dim=0).type_as(Xs)
        return edge_index, edge_attr

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def NormalizedSquareEuclideanDistance(self, x, y, eps=1e-6):
        ned = 0.5 * ((x - y).var() / (x.var() + y.var() + eps))
        return ned
