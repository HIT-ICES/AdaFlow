import os
import torch as th
import torch.nn as nn
import json
import dgl
import numpy as np
from dgl.data import DGLDataset
from clearml import Dataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
from collections import Counter
from torch_geometric.utils import add_self_loops

class ProgrammableWebDataset(DGLDataset):
    r""" ProgrammableWeb is is the largest online Web service registry. 
    This dataset can be used for service (bundle) recommendation, service label
    recommendation task. For more statics infos please view this dataset on Lab's
    ClearML server.
    
    Parameters
    ----------
    raw_dir : str
        Specifying the directory that will store the downloaded data
        or the directory that already stores the input data.
        Default: ~/.dgl/
    force_reload : bool
        Wether to reload the dataset. Default: False
    verbose : bool
        Wether to print out progress information. Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """
    _dataset_id = "48671720506142f09038527d83a98807"
    _mashup_num = 7668
    _api_num = 23469
    # _api_num = 1579
    save_dir = "./data/PW"
    save_path = "./data/PW"
    propensity_score = None

    def __init__(self, stage, raw_dir=None, force_reload=False, verbose=False, transform=None, t_mask=None):
        self.stage = stage
        self.t_mask = t_mask
        super(ProgrammableWebDataset, self).__init__(
            name="programmable", url=self._dataset_id, raw_dir=raw_dir, force_reload=force_reload,
            verbose=verbose, transform=transform)

    def download(self):
        if not os.path.exists(self.save_dir):
            dataset = Dataset.get(dataset_id=self._dataset_id)
            dataset.get_mutable_local_copy(target_folder=self.save_dir)
    
    def process(self):
        # NOTE(@Mingyi Liu): I only provide the procession for service recommendation task.
        invocation_records = json.load(
            open(os.path.join(self.raw_dir, "invocation.json"), encoding='utf8')
        )
        mashups, apis = invocation_records["Xs"], invocation_records["Ys"]
        timestamps, t_masks = invocation_records["time"], invocation_records["mask"]
        
        if self.t_mask is not None:
            t_masks = self.t_mask
        flat_apis = [api for api_list in apis for api in api_list]
        flat_apis = Counter(flat_apis)
        api_freqs = th.tensor([flat_apis[i] for i in range(self._api_num)])
        api_freqs += 1
        self.propensity_score = api_freqs / th.sum(api_freqs)

        g_labels = {"mashups": []}
        
        train_g_lists = []
        val_g_lists = []
        test_g_lists = []
        
        graph_all = dgl.graph(([], []), num_nodes=self._api_num)
        graph_all.ndata['freq'] = api_freqs
        node_ids = th.arange(self._api_num)
        src, dst = th.meshgrid(node_ids, node_ids)
        graph_all.add_edges(src.flatten(), dst.flatten())
        
        g_lists = []
        all_mashup, all_invoked_apis = [], []
        for (g_apis, mashup, mask) in zip(apis, mashups, t_masks):
            graph = graph_all.subgraph(g_apis)
            if mask == 0 and self.stage == 'train':
                g_labels["mashups"].append(mashup)
                g_lists.append(graph)
                all_mashup.append(mashup)
                all_invoked_apis.append(g_apis)
            elif mask == 1 and self.stage == 'val':
                g_labels["mashups"].append(mashup)
                g_lists.append(graph)
            elif mask == 2 and self.stage == 'test':
                g_labels["mashups"].append(mashup)
                g_lists.append(graph)
        g_labels["mashups"] = th.tensor(g_labels["mashups"])
        
        self.g_labels = g_labels
        self.g_lists = g_lists

        source_mashup, target_api = [], []
        for m, apis in zip(all_mashup, all_invoked_apis):
            for api in apis:
                source_mashup.append(m+self._api_num)
                source_mashup.append(api)
                target_api.append(api)
                target_api.append(m+self._api_num)
        source_mashup = th.LongTensor(source_mashup)
        target_api = th.LongTensor(target_api)
        self.ma_graph = th.stack([source_mashup, target_api], dim=0)
        self.ma_graph, _ = add_self_loops(self.ma_graph, num_nodes=self._api_num + self._mashup_num)
    
    def save(self):
        graph_path = os.path.join(self.save_path, f"pw-sr-{self.stage}.bin")        
        save_graphs(str(graph_path), self.g_lists, self.g_labels)

        propensity_score_path = os.path.join(self.save_path, 'propensity_score.pt')
        ma_graph_path = os.path.join(self.save_path, 'ma_graph.pt')
        if self.stage == 'train':
            th.save(self.propensity_score, propensity_score_path)
            th.save(self.ma_graph, ma_graph_path)
            print(ma_graph_path)

    
    def load(self):
        graph_path = os.path.join(self.save_path, f"pw-sr-{self.stage}.bin")
        g_lists, g_labels = load_graphs(str(graph_path))
        self.g_lists = g_lists
        self.g_labels = g_labels
    
    def __len__(self):
        return len(self.g_lists)

    def __getitem__(self, idx):
        return self.g_lists[idx], self.g_labels["mashups"][idx]