import torch.nn as nn
import torch
from typing import Dict, Tuple
import dgl.nn.pytorch as dglnn
from models.heterographconv import HeteroGraphConv
import dgl.function as fn

class HeteroGraphSAGE(nn.Module):
    def __init__(self,
                 input_dropout: float,
                 dropout: float,
                 hidden_dim: int,
                 feat_dict: Dict[Tuple[str, str, str],
                                 Tuple[int, int, int]],
                 task: str = 'claim'):
        super().__init__()
        self.feat_dict = feat_dict
        self.hidden_dim = hidden_dim
        self.task = task

        self.conv1 = HeteroGraphConv(
            {rel: dglnn.SAGEConv(in_feats=(feats[0], feats[1]),
                                 out_feats=hidden_dim,
                                 aggregator_type='lstm',
                                 feat_drop=input_dropout,
                                 activation=nn.GELU())
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.conv2 = HeteroGraphConv(
            {rel: dglnn.SAGEConv(in_feats=hidden_dim,
                                 out_feats=hidden_dim,
                                 aggregator_type='lstm',
                                 feat_drop=dropout,
                                 activation=nn.GELU())
             for rel, _ in feat_dict.items()},
            aggregate='sum')

        self.clf = nn.Sequential(
            # nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, 2) # classification
            nn.Linear(hidden_dim, 1)  # regression
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.w = dglnn.EdgeWeightNorm()


    def forward(self, blocks, h_dict: dict, eweight=None) -> dict:

        if eweight is None:
            h_dict = self.conv1(blocks[0], h_dict)

        else:
            h_dict = self.conv1(blocks[0], h_dict, mod_kwargs={
                c_etype: {'edge_weight': self.w(blocks[0][c_etype], eweight[c_etype])} for c_etype in blocks[0].canonical_etypes
            })

        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        h_dict = self.conv2(blocks[1], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        return self.clf(h_dict[self.task])

