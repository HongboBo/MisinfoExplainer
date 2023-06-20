from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam
from functools import partial
import numpy as np
from math import sqrt
import torch
from torch.autograd import grad as original_grad

# def new_grad(*args, **kwargs):
#     kwargs['allow_unused'] = True
#
#     return original_grad(*args, **kwargs)
#
# torch.autograd.grad = new_grad


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroGNNExplainer(nn.Module):
    def __init__(self, model, num,log=True):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.num = num
        self.log = log


    def forward_for_one_node_type(self, x, node_type, block, x_dict):
        #https://discuss.dgl.ai/t/captum-ai-explainability-for-heterogeneous-graphs/3135/8
        x_dict = x_dict.copy()
        x_dict[node_type] = x
        return self.model(block, x_dict)

    def forward_model(self, edge_masks, c_etype, block, x_dict, eweight):
        x_dict = x_dict.copy()
        eweight[c_etype] = edge_masks
        output_sum = sum(
            torch.sum(self.model.w(block[0][c_etype], eweight[c_etype])) for c_etype in block[0].canonical_etypes)
        return self.model(block, x_dict, eweight=eweight) + output_sum


    def explain_graph(self, block, feat, label, **kwargs):
        self.model = self.model.to(device)
        eweight = {}

        for c_etype in block[0].canonical_etypes:
            eweight[c_etype] = torch.ones(block[0].num_edges(c_etype)).to(device)
            eweight[c_etype].requires_grad = True

        pred_label = self.model(block, feat, eweight=eweight, **kwargs).squeeze()
        loss = F.mse_loss(
            input=pred_label,
            target=label.float(),
        )
        loss.backward()

        if self.log:
            pbar = tqdm(total=self.num)
            pbar.set_description('Explain graph')

        edge_masks = {}
        for c_etype in block[0].canonical_etypes:
            ig = IntegratedGradients(self.forward_model)
            e_m = ig.attribute(eweight[c_etype],
                               additional_forward_args=(c_etype, block, feat, eweight), internal_batch_size=eweight[c_etype].size(0), n_steps=50)
            edge_masks[c_etype] = e_m.sum()


        feat_masks = {}
        for ntype, feature in feat.items():
            ig = IntegratedGradients(partial(self.forward_for_one_node_type,
                                         node_type=ntype, block=block, x_dict=feat))
            f_m = ig.attribute(feature, internal_batch_size=feature.size(0), n_steps=50).abs()

            feat_masks[ntype] = f_m.sum(dim=0)

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_masks:
            feat_masks[node_type] = feat_masks[node_type].detach().squeeze()


        if self.log:
            pbar.close()

        for node_type in feat_masks:
            feat_masks[node_type] = feat_masks[node_type].abs().sigmoid().detach().squeeze()

        for c_etype in edge_masks:
            edge_masks[c_etype] = edge_masks[c_etype].abs().sigmoid().detach().squeeze()



        return feat_masks, edge_masks








