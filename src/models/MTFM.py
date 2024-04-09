from typing import Any, Optional, List

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from src.utils.metrics import Precision, NormalizedDCG, Recall, MILC
import time
import torch.nn as nn
import dgl
import numpy as np
import os
import csv


class MTFMModel(LightningModule):

    def __init__(self,
                lr = 1e-4,
                weight_decay: float = 1e-4,
                embed_dim = 300,
                max_doc_len = 50,
                dropout = 0.2,
                feature_dim = 8,
                num_kernel = 128,
                kernel_size = None,
                num_mashup = 4461,
                num_api = 945,
                feat_size = 128,
                frozen_api_embeds = True,
                top_k = 5,
                metric_path = None,
                seed = None
                ):
        super(MTFMModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.embed_dim = embed_dim
        self.max_doc_len = max_doc_len
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.num_mashup = num_mashup
        self.num_api = num_api
        self.feat_size = feat_size
        self.frozen_api_embeds = frozen_api_embeds
        self.top_k = top_k
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metric_path = metric_path
        self.seed = seed

        # [M, D]
        self.register_buffer("mashup_embeds", torch.from_numpy(
            np.load("data/PW/mashup_dev_embeds.npz", allow_pickle=True)['arr_0']
        ).float())

        # [A, D]
        if frozen_api_embeds:
            self.register_buffer("api_embeds", torch.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float())
        else:
            self.register_parameter("api_embeds", nn.Parameter(torch.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float()))
        self.mashup_feat_size = self.mashup_embeds.size(1)  # type: ignore[misc]
        self.api_feat_size = self.api_embeds.size(1)  # type: ignore[misc]

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

        metric_file = open(os.path.join(self.metric_path, str(self.seed) + '.csv'), 'w', encoding='utf-8', newline="")
        self.metric_csv = csv.writer(metric_file)
        self.metric_csv.writerow(['Epoch', 'idx_0', 'idx_1'])

        self.idx_0 = 0
        self.idx_1 = 0
        self.epoch = -1

    def _build_layers(self):
        self.m_fc = nn.Linear(self.mashup_feat_size, self.feat_size)  # [M, D] -> [M, F]
        self.a_fc = nn.Linear(self.api_feat_size, self.feat_size)  # [A, D] -> [A, F]
        self.sc_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.feat_size, out_channels=self.num_kernel, kernel_size=h),
                # 128, 128, [2, 3, 4, 5]
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_doc_len - h + 1)
            )
            for h in self.kernel_size
        ])
        self.sc_fcl = nn.Linear(in_features=self.num_kernel * len(self.kernel_size),
                                out_features=self.num_api)  # 256 * 4 = 1024, num_api

        self.fic_fc = nn.Linear(in_features=self.num_kernel * len(self.kernel_size),
                                out_features=self.feature_dim)  # 1024, 8
        self.fic_api_feature_embedding = nn.Parameter(torch.rand(self.feature_dim, self.num_api))  # [8, num_api]
        self.fic_mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),  # 16, 8
            nn.Linear(self.feature_dim, 1),  # 8, 1
            nn.Tanh()
        )
        self.fic_fcl = nn.Linear(self.num_api * 2, self.num_api)  # [num_api * 2, num_api]

        self.fusion_layer = nn.Linear(self.num_api * 2, self.num_api)  # [num_api * 2, num_api ]

        self.api_task_layer = nn.Linear(self.num_api, self.num_api)
        # self.category_task_layer = nn.Linear(self.num_api, self.num_category)

        self.dropout = nn.Dropout(self.dropout)  # 0.2
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.epoch += 1
            self.idx_0 = self.P5_val_0.compute()
            self.idx_1 = self.P5_val_1.compute()
            if self.epoch != -1:
                self.metric_csv.writerow([
                    self.epoch, self.idx_0.cpu().item(), self.idx_1.cpu().item()
                ])

    def on_train_start(self):
        nn.init.kaiming_normal_(self.fic_api_feature_embedding)
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

        # semantic component
        start_time = time.time()
        mashup_des = mashup_feats[graph_labels]
        embed = mashup_des.unsqueeze(dim=-1)
        embed = embed.expand(embed.shape[0], embed.shape[1], 10)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_fcl(e)

        # feature interaction component
        u_sc_trans = self.fic_fc(e)
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)
        u_concate = []
        for u_sc_single in u_sc_trans:
            u_concate_single = torch.cat((u_sc_single.repeat(self.fic_api_feature_embedding.size(1), 1), self.fic_api_feature_embedding.t()), dim=1)
            u_concate.append(self.fic_mlp(u_concate_single).squeeze())
        u_mlp = torch.cat(u_concate).view(u_mm.size(0), -1)
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp),
                                       dim=1))
        u_fic = self.tanh(u_fic)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic),
                                            dim=1))

        # dropout
        u_mmf = self.dropout(u_mmf)

        # api-specific task layer
        pred = self.api_task_layer(u_mmf).to("cuda:0")
        pred = self.logistic(pred)
        self.train_escape += time.time() - start_time

        target = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.zeros(self.num_api).float()
            masks[pos_nodes] = 1
            target.append(masks)
        start_time = time.time()
        target = torch.stack(target, dim=0).float().to("cuda:0")
        loss = self.criterion(pred, target)
        self.train_escape += time.time() - start_time
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        batch_graph, graph_labels = batch
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        graphs = dgl.unbatch(batch_graph)
        # semantic component
        mashup_des = mashup_feats[graph_labels]
        embed = mashup_des.unsqueeze(dim=-1)
        embed = embed.expand(embed.shape[0], embed.shape[1], 10)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_fcl(e)

        # feature interaction component
        u_sc_trans = self.fic_fc(e)
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)
        u_concate = []
        for u_sc_single in u_sc_trans:
            u_concate_single = torch.cat(
                (u_sc_single.repeat(self.fic_api_feature_embedding.size(1), 1), self.fic_api_feature_embedding.t()),
                dim=1
            )
            u_concate.append(self.fic_mlp(u_concate_single).squeeze())
        u_mlp = torch.cat(u_concate).view(u_mm.size(0), -1)
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp), dim=1))
        u_fic = self.tanh(u_fic)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))

        # dropout
        u_mmf = self.dropout(u_mmf)

        # api-specific task layer
        pred = self.api_task_layer(u_mmf).to("cuda:0")
        pred = self.logistic(pred)

        target = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.zeros(self.num_api).float()
            masks[pos_nodes] = 1
            target.append(masks)
        target = torch.stack(target, dim=0).float().to("cuda:0")

        if not self.trainer.sanity_checking:
            if loader_idx == 0:
                self.P5_val_0.update(pred, target)
                self.PSP5_val_0.update(pred, target)
                self.R5_val_0.update(pred, target)
                self.PSR5_val_0.update(pred, target)
                self.nDCG5_val_0.update(pred, target)
                self.PSnDCG5_val_0.update(pred, target)
                self.log("val/P@5", self.P5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_0, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSR@5", self.PSR5_val_0, on_step=False, on_epoch=True)
                self.log("val/nDCG@5", self.nDCG5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSnDCG@5", self.PSnDCG5_val_0, on_step=False, on_epoch=True)
            else:
                self.P5_val_1.update(pred, target)
                self.PSP5_val_1.update(pred, target)
                self.R5_val_1.update(pred, target)
                self.PSR5_val_1.update(pred, target)
                self.nDCG5_val_1.update(pred, target)
                self.PSnDCG5_val_1.update(pred, target)
                self.log("val/P@5", self.P5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_1, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSR@5", self.PSR5_val_1, on_step=False, on_epoch=True)
                self.log("val/nDCG@5", self.nDCG5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSnDCG@5", self.PSnDCG5_val_1, on_step=False, on_epoch=True)


    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        batch_graph, graph_labels = batch
        start_time = time.time()
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        self.test_escape += time.time() - start_time
        graphs = dgl.unbatch(batch_graph)
        # semantic component
        start_time = time.time()
        mashup_des = mashup_feats[graph_labels]
        embed = mashup_des.unsqueeze(dim=-1)
        embed = embed.expand(embed.shape[0], embed.shape[1], 10)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_fcl(e)

        # feature interaction component
        u_sc_trans = self.fic_fc(e)
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)
        u_concate = []
        for u_sc_single in u_sc_trans:
            u_concate_single = torch.cat((u_sc_single.repeat(self.fic_api_feature_embedding.size(1), 1), self.fic_api_feature_embedding.t()), dim=1 )
            u_concate.append(self.fic_mlp(u_concate_single).squeeze())
        u_mlp = torch.cat(u_concate).view(u_mm.size(0), -1)
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp), dim=1))
        u_fic = self.tanh(u_fic)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))

        # dropout
        u_mmf = self.dropout(u_mmf)

        # api-specific task layer
        preds = self.api_task_layer(u_mmf).to("cuda:0")
        preds = self.logistic(preds)
        self.test_escape += time.time() - start_time

        logits = []
        for graph in graphs:
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.zeros(self.num_api).float()
            masks[pos_nodes] = 1
            logits.append(masks)
        logits = torch.stack(logits, dim=0).float().to("cuda:0")
        self.P5_test.update(preds, logits)
        self.PSP5_test.update(preds, logits)
        self.R5_test.update(preds, logits)
        self.PSR5_test.update(preds, logits)
        self.nDCG5_test.update(preds, logits)
        self.PSnDCG5_test.update(preds, logits)
        self.log("test/P@5", self.P5_test, on_step=False, on_epoch=True)
        self.log("test/PSP@5", self.PSP5_test, on_step=False, on_epoch=True)
        self.log("test/R@5", self.R5_test, on_step=False, on_epoch=True)
        self.log("test/PSR@5", self.PSR5_test, on_step=False, on_epoch=True)
        self.log("test/nDCG@5", self.nDCG5_test, on_step=False, on_epoch=True)
        self.log("test/PSnDCG@5", self.PSnDCG5_test, on_step=False, on_epoch=True)
        self.log("test/escape", self.test_escape, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
