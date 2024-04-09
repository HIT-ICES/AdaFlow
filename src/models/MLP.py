import os

import pytorch_lightning as pl
import numpy as np
import torch as th
import torch.nn as nn
import time
import dgl
import csv
import json
import pandas as pd

from src.utils.metrics import Precision, NormalizedDCG, Recall, MILC

class MLP(pl.LightningModule):
    
    def __init__(self,
                 num_api,
                 num_mashups,
                 feat_size,
                 lr,
                 weight_decay,
                 frozen_api_embeds=False,
                 top_k=5,
                 metric_path=None,
                 seed = None,
                 save_path = None,
                 ) -> None:
        super().__init__()
        self.save_path = save_path
        self.num_api = num_api
        self.num_mashups = num_mashups
        self.feat_size = feat_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = th.nn.BCEWithLogitsLoss()
        self.top_k = top_k
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
        
        
        self._build_layers()

        self.idx_0 = 0
        self.idx_1 = 0
        self.epoch = 0


    def _build_layers(self):
        self.m_fc = nn.Linear(self.mashup_feat_size, self.feat_size)
        self.a_fc = nn.Linear(self.api_feat_size, self.feat_size)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.feat_size * 2, self.feat_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_size, 1)
        )

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

    def on_train_epoch_start(self) -> None:
        self.train_escape = 0
    def training_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        start_time = time.time()
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        self.train_escape += time.time() - start_time
        graphs = dgl.unbatch(batch_graph)
        
        mashups, apis, labels = [], [], []
        for graph, mashup in zip(graphs, graph_labels):
            pos_nodes = graph.ndata[dgl.NID]
            masks = th.ones(self.num_api).type_as(mashup).float()
            masks[pos_nodes] = 0 # type: ignore
            neg_nodes = th.multinomial(masks, num_samples=graph.num_nodes() * 5, replacement=False)
            
            mashups.extend([mashup.cpu().item()] * (graph.num_nodes() * 6))
            labels.extend([1] * graph.num_nodes())
            labels.extend([0] * (graph.num_nodes() * 5))
            apis.extend(pos_nodes.cpu().numpy().tolist())
            apis.extend(neg_nodes.cpu().numpy().tolist())
        mashups = th.tensor(mashups).type_as(mashup_feats).long()
        apis = th.tensor(apis).type_as(mashup_feats).long()
        labels = th.tensor(labels).type_as(mashup_feats).float()
        start_time = time.time()
        in_feats = th.cat([mashup_feats[mashups], api_feats[apis]], dim=1)
        predicts = self.mlp1(in_feats).view(-1)
        
        loss = self.criterion(predicts, labels)
        self.train_escape += time.time() - start_time
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)
        return loss 
    

    def validation_step(self, batch, batch_idx, loader_idx):
        batch_graph, graph_labels = batch
        
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        graphs = dgl.unbatch(batch_graph)
        
        preds, logits = [], []
        for graph, mashup in zip(graphs, graph_labels):
            mashup_embeds = mashup_feats[mashup].unsqueeze(0).repeat(self.num_api, 1)
            in_feats = th.cat([mashup_embeds, api_feats], dim=1)
            pred = self.mlp1(in_feats).view(1, -1) # [1, N]
            preds.append(th.sigmoid(pred))
            logist_onehot = th.zeros(self.num_api).type_as(pred).float()
            logist_onehot[graph.ndata[dgl.NID]] = 1.
            logits.append(logist_onehot.view(1, -1))
        p_t = th.cat(preds, dim=0)
        l_t = th.cat(logits, dim=0)
        if not self.trainer.sanity_checking:
            if loader_idx == 0:
                self.P5_val_0.update(p_t, l_t)
                self.PSP5_val_0.update(p_t, l_t)
                self.R5_val_0.update(p_t, l_t)
                self.log("val/P@5", self.P5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_0, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_0, on_step=False, on_epoch=True)
            else:
                self.P5_val_1.update(p_t, l_t)
                self.PSP5_val_1.update(p_t, l_t)
                self.R5_val_1.update(p_t, l_t)
                self.log("val/P@5", self.P5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_1, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_1, on_step=False, on_epoch=True)

    def on_test_start(self) -> None:
        self.mashup_real_preds = []

    def test_step(self, batch, batch_idx):
        example_mashup = {}
        batch_graph, graph_labels = batch
        start_time = time.time()
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        graphs = dgl.unbatch(batch_graph)

        preds, logits = [], []
        for graph, mashup in zip(graphs, graph_labels):
            mashup_embeds = mashup_feats[mashup].unsqueeze(0).repeat(self.num_api, 1)
            in_feats = th.cat([mashup_embeds, api_feats], dim=1)
            pred = self.mlp1(in_feats).view(1, -1) # [1, N]
            top_k, indices = th.topk(pred, 5)
            preds.append(th.sigmoid(pred))
            logist_onehot = th.zeros(self.num_api).type_as(pred).float()
            logist_onehot[graph.ndata[dgl.NID]] = 1.
            logits.append(logist_onehot.view(1, -1))
        preds = th.cat(preds, dim=0)
        self.test_escape += time.time() - start_time
        logits = th.cat(logits, dim=0)
        self.P5_test.update(preds, logits)
        self.PSP5_test.update(preds, logits)
        self.R5_test.update(preds, logits)
        self.log("test/P@5", self.P5_test, on_step=False, on_epoch=True)
        self.log("test/PSP@5", self.PSP5_test, on_step=False, on_epoch=True)
        self.log("test/R@5", self.R5_test, on_step=False, on_epoch=True)

    def _update_after_test(self, recommended_apis, api_feats, mashup):
        pass
    
    def configure_optimizers(self):
        return th.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
            
      
            
        