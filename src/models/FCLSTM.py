import imp
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import negative, nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
from src.utils.metrics import Precision, NormalizedDCG
from sklearn.svm import SVC
# from src.utils.semantictgap import semantic_gap_evaluation

class FCLSTM(pl.LightningModule):
    def __init__(self, 
                 mashup_embed_path, 
                 api_embed_path,
                 tags_embed_path,
                 lr: float = 1e-3,
                 semantic_path: str = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.register_buffer('mashup_embeds', torch.from_numpy(np.load(mashup_embed_path, allow_pickle=True)).float())
        self.register_parameter('api_embeds', Parameter(torch.from_numpy(np.load(api_embed_path, allow_pickle=True)).float()))
        self.register_parameter('tags_embeds', Parameter(torch.from_numpy(np.load(tags_embed_path, allow_pickle=True)).float()))

        self.mashup_embed_channels = self.mashup_embeds.size(1)
        self.api_embed_channels = self.api_embeds.size(1)
        self.num_api = self.api_embeds.size(0)

        # self.semantic_path = semantic_path

        self.lr = lr

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        #
        self.P5 = Precision(top_k=5)
        self.nDCG5 = NormalizedDCG(top_k=5)
        self.P5_val = Precision(top_k=5)
        self.nDCG5_val = NormalizedDCG(top_k=5)
    
    def format_samples(self, Xs, Ys):
        embeddings_mashup, invokded_apis_embeds, uninvoked_apis_embeds = [], [], []
        mashup_tag_embedds, positive_tag_embedds, negative_tag_embedds = [], [], []

        for X, Y in zip(Xs, Ys):
            mashup_tags = self.mashup_invoke_tags[X]    
            mashup_tag_embed = torch.mean(self.tags_embeds[mashup_tags == 1], dim=0)

            invoked_apis = Y.nonzero(as_tuple=True)[0]
            uninvoked_apis = torch.multinomial(1.-Y.float(), num_samples=len(invoked_apis), replacement=True)
            for api in invoked_apis:
                mashup_tag_embedds.append(mashup_tag_embed)
                embeddings_mashup.append(self.mashup_embeds[X])
                invokded_apis_embeds.append(self.api_embeds[api])
                invokded_apis_tag_embed = torch.mean(self.tags_embeds[self.api_invoke_tags[api] == 1], dim=0)
                positive_tag_embedds.append(invokded_apis_tag_embed)
            for api in uninvoked_apis:
                uninvoked_apis_embeds.append(self.api_embeds[api])
                uninvoked_apis_tag_embed = torch.mean(self.tags_embeds[self.api_invoke_tags[api] == 1], dim=0)
                negative_tag_embedds.append(uninvoked_apis_tag_embed)

        embeddings_mashup = torch.stack(embeddings_mashup)
        mashup_tag_embedds = torch.stack(mashup_tag_embedds)
        invokded_apis_embeds = torch.stack(invokded_apis_embeds)
        positive_tag_embedds = torch.stack(positive_tag_embedds)
        uninvoked_apis_embeds = torch.stack(uninvoked_apis_embeds)
        negative_tag_embedds = torch.stack(negative_tag_embedds)

        return embeddings_mashup, mashup_tag_embedds, invokded_apis_embeds, positive_tag_embedds, uninvoked_apis_embeds, negative_tag_embedds
    
    def on_train_start(self) -> None:
        self.propensity_score = self.trainer.datamodule.propensity_score
        self.propensity_score = self.propensity_score.to(self.device)

        self.api_invoke_tags = self.trainer.datamodule.api_invoke_tags
        self.api_invoke_tags = self.api_invoke_tags.to(self.device)
        self.mashup_invoke_tags = self.trainer.datamodule.mashup_invoke_tags
        self.mashup_invoke_tags = self.mashup_invoke_tags.to(self.device)

        self.PSP5 = Precision(top_k=5, propensity_score=self.propensity_score)
        self.PSDCG5 = NormalizedDCG(top_k=5, propensity_score=self.propensity_score)
        self.PSP5_val = Precision(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.PSDCG5_val = NormalizedDCG(top_k=5, propensity_score=self.propensity_score).to(self.device)
        self.escape = 0
        
        
    
    def on_train_epoch_start(self) -> None:
        self.train_escape = 0 
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()
        Xs, Ys, _ = batch # Xs: [B], Ys: [B, C]
        # prepare supported format
        embeddings_mashup, mashup_tag_embedds, positive_embedds, positive_tag_embedds, negative_embedds, negative_tag_embedds = self.format_samples(Xs, Ys)
        mashup_embeds = torch.cat([embeddings_mashup, mashup_tag_embedds], dim=1)
        positive_embedds = torch.cat([positive_embedds, positive_tag_embedds], dim=1)
        negative_embedds = torch.cat([negative_embedds, negative_tag_embedds], dim=1)
        pos_loss = 1 - 2 * self.cos(mashup_embeds, positive_embedds)
        neg_loss = 1 + 2 * self.cos(mashup_embeds, negative_embedds)
        loss = F.relu(pos_loss).sum() + F.relu(neg_loss).sum()

        # now feed to mlp
        self.log("train/loss", loss)
        self.train_escape += time.time() - start_time
        self.log("train/escape", self.train_escape, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        Xs, Ys, _ = batch # Xs: [B], Ys: [B, C]
        if self.trainer.sanity_checking:
            return
            Xs, Ys, _ = batch # Xs: [B], Ys: [B, C]
        # prepare supported format
        embeddings_mashup, mashup_tag_embedds, positive_embedds, positive_tag_embedds, negative_embedds, negative_tag_embedds = self.format_samples(Xs, Ys)
        mashup_embeds = torch.cat([embeddings_mashup, mashup_tag_embedds], dim=1)
        positive_embedds = torch.cat([positive_embedds, positive_tag_embedds], dim=1)
        negative_embedds = torch.cat([negative_embedds, negative_tag_embedds], dim=1)
        pos_loss = 0.1 - 2 * self.cos(mashup_embeds, positive_embedds)
        neg_loss = 0.1 + 2 * self.cos(mashup_embeds, negative_embedds)
        loss = F.relu(pos_loss).sum() + F.relu(neg_loss).sum()
        self.log("val/loss", loss)

    
    def on_test_start(self) -> None:
        self.test_escap = 0
        
    def test_step(self, batch, batch_idx):
        start_time = time.time()
        Xs, Ys, _ = batch # Xs: [B], Ys: [B, C]
        if self.trainer.sanity_checking:
            return
        preds = []
        api_feature = []
        # All api embeddings and corresponding tags embeddings
        for api_embeds, tags in zip(self.api_embeds, self.api_invoke_tags):
            api_tag_embeds = torch.mean(self.tags_embeds[tags == 1], dim=0)
            api_feature.append(torch.cat([api_embeds, api_tag_embeds], dim=0))
        api_feature = torch.stack(api_feature)  # [N, D] 

        for mashup in Xs:
            mashup_embed = self.mashup_embeds[mashup]
            mashup_tags = self.mashup_invoke_tags[mashup]
            mashup_tag_embeds = torch.mean(self.tags_embeds[mashup_tags == 1], dim=0)
            mashup_feature = torch.cat([mashup_embed, mashup_tag_embeds], dim=0)
            mashup_feature = mashup_feature.view(1, -1).repeat(api_feature.size(0), 1)
            pred = self.cos(mashup_feature, api_feature)
            preds.append(pred)
        preds = torch.stack(preds)
        self.test_escap += time.time() - start_time
        self.P5.update(preds, Ys)
        self.nDCG5.update(preds, Ys)
        self.PSP5.update(preds, Ys)
        self.PSDCG5.update(preds, Ys)
        self.log("test/escape", self.test_escap, on_step=False, on_epoch=True)
        self.log("test/P@5", self.P5.compute(), on_step=False, on_epoch=True)
        self.log("test/nDCG@5", self.nDCG5.compute(), on_step=False, on_epoch=True)
        self.log("test/PS@5", self.PSP5.compute(), on_step=False, on_epoch=True)
        self.log("test/PSDCG@5", self.PSDCG5.compute(), on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        mashup_embeds = self.mashup_embeds.detach().cpu().numpy()
        api_embedds = self.api_embeds.detach().cpu().numpy()
        semantic_gaps = semantic_gap_evaluation(self.semantic_path, api_embedds, mashup_embeds)  
        self.logger.log_metrics({"semantic_gaps": semantic_gaps})

    
    def configure_optimizers(self):
        return torch.optim.SGD(
            params=self.parameters(), lr=self.lr
        )

        
        
        