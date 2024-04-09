import numpy as np
import dgl
import torch as th
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.datamodules.dataset.ProgrammableWebDataset import ProgrammableWebDataset
import os

class PWDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size = 64, raw_dir="./data/PW/", force_reload=False, verbose=False, transform=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = ProgrammableWebDataset('train', raw_dir, force_reload, verbose, transform)
        self.val_dataset = ProgrammableWebDataset('val', raw_dir, force_reload, verbose, transform)
        self.test_dataset = ProgrammableWebDataset('test', raw_dir, force_reload, verbose, transform)
        self.batch_size = batch_size
        self.propensity_score_path = os.path.join(raw_dir, 'propensity_score.pt')
        self.ma_graph_path = os.path.join(raw_dir, 'ma_graph.pt')
        self.propensity_score = None
        self.ma_graph = None

    def setup(self, stage = None):
        if os.path.exists(self.propensity_score_path):
            self.propensity_score = th.load(self.propensity_score_path)
        if os.path.exists(self.ma_graph_path):
            self.ma_graph = th.load(self.ma_graph_path)

    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)
        # return None
    def val_dataloader(self):
        return [dgl.dataloading.GraphDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4),
                dgl.dataloading.GraphDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)]
        # return None
    def test_dataloader(self):
        return dgl.dataloading.GraphDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)
        # return None
        