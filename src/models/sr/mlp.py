import os
import numpy as np
import pytorch_lightning as pl

from src.utils.metrics import Precision
from torchmetrics import Precision
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MLP(pl.LightningModule):
    r""" MLP for mashup creation. An simple baseline for service recommendation.
    
    Parameters
    ----------
    mashup_embeds : str
        Pre-trained mashup embeddings file path
        
    api_embeds : str
        Pre-trained api embeddings file path
    
    negative_sample : int
        How many negative samples are generated for each positive sample.
        
    lr : float
        learning rate. Default `0.001`
        
    
    weight_decay : float
        Default `1e-4`
    """
    def __init__(self, mashup_embeds: str, api_embeds: str, negative_sample: int = 5, lr: float = 1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.register_buffer('mashup_embeds', th.from_numpy(np.load(mashup_embeds, allow_pickle=True)).float())
        self.register_buffer('api_embeds', th.from_numpy(np.load(api_embeds, allow_pickle=True)).float())
        
        self.m_feats = self.mashup_embeds.size(1)
        self.a_feats = self.api_embeds.size(1)
        self.num_api = self.api_embeds.size(0)
        
        self.negative_sample = negative_sample
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.P5 = Precision(top_k=5)
        self.P5_val = Precision(top_k=5)
        
        self._build_layers()

    def _build_layers(self):
        self.mlp = nn.Sequential(
            nn.Linear(self.m_feats + self.a_feats, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def training_step(self, batch, batch_idx):
        pass
        
    
        