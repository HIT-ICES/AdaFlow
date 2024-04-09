import torch
from torch import Tensor
import numpy as np
import pytorch_lightning as pl
from torch import nn
import time
from src.utils.metrics import Precision, NormalizedDCG, Recall, MILC
import dgl
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor

class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{e_{j,i}}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j

    Args:
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized via symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    def __init__(self, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight,
                              size=None)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class coACN(pl.LightningModule):
    def __init__(self,
                 negative_sample=5,
                 lr: float = 1e-3,
                 frozen_api_embeds=True,
                 feat_size=128,
                 num_api=945,
                 num_mashup=4461,
                 weight_decay=0.0001,
                 metric_path=None,
                 seed=None,
                 save_path=None,
                 ) -> None:
        print("debug!!!")
        super().__init__()
        print("debug!!!")
        self.save_hyperparameters()

        self.register_buffer("mashup_embeds", torch.from_numpy(
            np.load("data/PW/mashup_dev_embeds.npz", allow_pickle=True)['arr_0']
        ).float())
        if frozen_api_embeds:
            self.register_buffer("api_embeds", torch.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float())
        else:
            self.register_parameter("api_embeds", nn.Parameter(torch.from_numpy(
                np.load("data/PW/api_dev_embeds.npz", allow_pickle=True)['arr_0']
            ).float()))

        self.register_buffer("tag_embeds", torch.from_numpy(
            np.load("data/PW/tag_embeds.npz", allow_pickle=True)['arr_0']
        ).float())
        self.save_path = save_path
        self.mashup_embed_channels = self.mashup_embeds.size(1)
        self.api_embed_channels = self.api_embeds.size(1)
        self.tag_embed_channels = self.tag_embeds.size(1)
        self.mashup_feat_size = self.mashup_embeds.size(1)
        self.api_feat_size = self.api_embeds.size(1)
        self.tag_feat_size = self.tag_embeds.size(1)
        self.num_api = num_api
        self.feat_size = feat_size
        self.num_api = num_api
        self.num_mashup = num_mashup
        self.weight_decay = weight_decay
        self.metric_path = metric_path
        self.seed = seed


        self.negative_sample = negative_sample
        self.lr = lr

        self.criterion = torch.nn.BCEWithLogitsLoss()

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
        self.attn = nn.MultiheadAttention(self.feat_size, num_heads=1, batch_first=True)
        self.lin_1 = nn.Linear(self.feat_size, self.feat_size)
        self.lin_3 = nn.Linear(self.feat_size, self.feat_size)

        self.lightGcn_1 = LGConv()
        self.lightGcn_2 = LGConv()

        self.m_fc = nn.Linear(self.mashup_feat_size, self.feat_size)
        self.a_fc = nn.Linear(self.api_feat_size, self.feat_size)
        self.t_fc = nn.Linear(self.tag_feat_size, self.feat_size)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.idx_0 = self.P5_val_0.compute()
            self.idx_1 = self.P5_val_1.compute()
            self.metric_csv.writerow([
                self.epoch, self.idx_0.cpu().item(), self.idx_1.cpu().item()
            ])
            self.epoch += 1

    def on_train_start(self) -> None:
        self.propensity_score = self.trainer.datamodule.propensity_score
        self.propensity_score = self.propensity_score.to(self.device)
        self.ma_graph = self.trainer.datamodule.ma_graph.to(self.device)

        self.PSP5_test = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_test = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_test = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_0 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_0 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_0 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSR5_val_1 = Recall(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSP5_val_1 = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSnDCG5_val_1 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)

        self.escape = 0

    def on_train_epoch_start(self) -> None:
        self.train_escape = 0
    def training_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)

        start_time = time.time()
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        tag_feats = self.t_fc(self.tag_embeds)

        api_feats = self.lin_1(api_feats)
        gx = torch.cat([api_feats, mashup_feats], dim=0)
        # structure information extraction
        gx = self.lightGcn_1(gx, self.ma_graph)
        gx = 0.5 * gx + 0.5 * self.lightGcn_2(gx, self.ma_graph)

        # service domain Enhancement
        mashup_embeddings = mashup_feats[graph_labels]
        query = mashup_embeddings.unsqueeze(1)
        key = self.lin_3(tag_feats)
        key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        out, _ = self.attn(query, key, key)
        out = out.squeeze(1)
        out = 0.4 * out + 0.6 * mashup_embeddings
        self.train_escape += time.time() - start_time
        # format_sample
        mashups, apis, labels = [], [], []
        for o, graph, mashup in zip(out, graphs, graph_labels):
            pos_nodes = graph.ndata[dgl.NID]
            masks = torch.ones(self.num_api).type_as(mashup).float()
            masks[pos_nodes] = 0  # type: ignore
            neg_nodes = torch.multinomial(masks, num_samples=graph.num_nodes() * 5, replacement=False)

            outs = torch.unsqueeze(o, dim=0)
            outs = np.repeat(outs.cpu().detach(), (graph.num_nodes() * 6), axis=0)
            mashups.extend(outs)
            labels.extend([1] * graph.num_nodes())
            labels.extend([0] * (graph.num_nodes() * 5))
            apis.extend(pos_nodes.cpu().numpy().tolist())
            apis.extend(neg_nodes.cpu().numpy().tolist())
        start_time = time.time()
        mashups = torch.stack(mashups).to(self.device)
        apis = torch.tensor(apis).type_as(mashup_feats).long()
        labels = torch.tensor(labels).type_as(mashup_feats).float()

        O_pos = gx[apis]
        preds = torch.sum(O_pos * mashups, dim=1)

        loss = self.criterion(preds, labels)
        self.train_escape += time.time() - start_time
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.P5 = []

    def validation_step(self, batch, batch_idx, loader_idx):
        if self.trainer.sanity_checking:
            return

        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)

        # 降维
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        tag_feats = self.t_fc(self.tag_embeds)

        api_feats = self.lin_1(api_feats)
        gx = torch.cat([api_feats, mashup_feats], dim=0)
        # structure information extraction
        gx = self.lightGcn_1(gx, self.ma_graph)
        gx = 0.5 * gx + 0.5 * self.lightGcn_2(gx, self.ma_graph)

        mashup_embeddings = mashup_feats[graph_labels]
        query = mashup_embeddings.unsqueeze(1)
        key = self.lin_3(tag_feats)
        key = key.unsqueeze(0).repeat(query.size(0), 1, 1, )
        out, _ = self.attn(query, key, key)
        out = out.squeeze(1)
        out = 0.4 * out + 0.6 * mashup_embeddings

        preds, logits = [], []
        for mashup_embeds in out:
            api_embeds = gx[:self.num_api]
            mashup_embeds = mashup_embeds.unsqueeze(0).repeat(self.num_api, 1)
            pred = torch.sigmoid(torch.sum(api_embeds * mashup_embeds, dim=1))
            preds.append(pred)
        for graph in graphs:
            logist_onehot = torch.zeros(self.num_api).float()
            logist_onehot[graph.ndata[dgl.NID]] = 1.
            logits.append(logist_onehot)
        preds = torch.stack(preds).to(self.device)
        logits = torch.stack(logits).to(self.device)

        if not self.trainer.sanity_checking:
            if loader_idx == 0:
                self.P5_val_0.update(preds, logits)
                self.PSP5_val_0.update(preds, logits)
                self.R5_val_0.update(preds, logits)
                self.log("val/P@5", self.P5_val_0, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_0, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_0, on_step=False, on_epoch=True)
            else:
                self.P5_val_1.update(preds, logits)
                self.PSP5_val_1.update(preds, logits)
                self.R5_val_1.update(preds, logits)
                self.log("val/P@5", self.P5_val_1, on_step=False, on_epoch=True)
                self.log("val/PSP@5", self.PSP5_val_1, on_step=False, on_epoch=True)
                self.log("val/R@5", self.R5_val_1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        batch_graph, graph_labels = batch
        graphs = dgl.unbatch(batch_graph)

        start_time = time.time()
        mashup_feats = self.m_fc(self.mashup_embeds)
        api_feats = self.a_fc(self.api_embeds)
        tag_feats = self.t_fc(self.tag_embeds)

        api_feats = self.lin_1(api_feats)
        gx = torch.cat([api_feats, mashup_feats], dim=0)
        # structure information extraction
        gx = self.lightGcn_1(gx, self.ma_graph)
        gx = 0.5 * gx + 0.5 * self.lightGcn_2(gx, self.ma_graph)

        mashup_embeddings = mashup_feats[graph_labels]
        query = mashup_embeddings.unsqueeze(1)
        key = self.lin_3(tag_feats)
        key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        out, _ = self.attn(query, key, key)
        out = out.squeeze(1)
        out = 0.4 * out + 0.6 * mashup_embeddings

        preds, logits = [], []
        for mashup_embeds, graph_label in out, graph_labels:
            api_embedds = gx[:self.num_api]
            mashup_embeds = mashup_embeds.unsqueeze(0).repeat(self.num_api, 1)
            pred = torch.sigmoid(torch.sum(api_embedds * mashup_embeds, dim=1))
            top_k, indices = torch.topk(pred, 5)
            preds.append(pred)
        for graph in graphs:
            logist_onehot = torch.zeros(self.num_api).float()
            logist_onehot[graph.ndata[dgl.NID]] = 1.
            logits.append(logist_onehot)
        preds = torch.stack(preds).to(self.device)
        logits = torch.stack(logits).to(self.device)
        self.test_escape += time.time() - start_time

        self.P5_test.update(preds, logits)
        self.PSP5_test.update(preds, logits)
        self.R5_test.update(preds, logits)
        self.log("test/P@5", self.P5_test, on_step=False, on_epoch=True)
        self.log("test/PSP@5", self.PSP5_test, on_step=False, on_epoch=True)
        self.log("test/R@5", self.R5_test, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def format_samples(self, Xs, Ys):
        embeddings_mashup, embeddings_apis, labels = [], [], []
        for X, Y in zip(Xs, Ys):
            invoked_apis = Y.nonzero(as_tuple=True)[0]
            uninvoked_apis = torch.multinomial(1. - Y.float(), num_samples=len(invoked_apis) * self.negative_sample,
                                               replacement=False)
            for api in invoked_apis:
                embeddings_mashup.append(X)
                embeddings_apis.append(api)
                labels.append(1)
            for api in uninvoked_apis:
                embeddings_mashup.append(X)
                embeddings_apis.append(api)
                labels.append(0)
        embeddings_mashup = torch.stack(embeddings_mashup)
        embeddings_apis = torch.stack(embeddings_apis)
        labels = torch.tensor(labels).type_as(Ys)
        return embeddings_mashup, embeddings_apis, labels
