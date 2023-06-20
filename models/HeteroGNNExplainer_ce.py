from torch import nn
from math import sqrt
import torch
from dgl.base import NID, EID
from dgl.subgraph import khop_in_subgraph
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroGNNExplainer(nn.Module):
    def __init__(self,
                 model,
                 num_hops,
                 lr=0.01,
                 num_epochs=100,
                 *,
                 alpha1=0.005,
                 alpha2=1.0,
                 beta1=1.0,
                 beta2=0.1,
                 log=True):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        feat_masks = {}
        std = 0.1
        for node_type, feature in feat.items():
            _, feat_size = feature.size()
            feat_masks[node_type] = nn.Parameter(torch.randn(1, feat_size, device=device) * std)


        edge_masks = {}
        # print(type(graph))

        for canonical_etype in graph[0].canonical_etypes:
            src_num_nodes = graph[0].num_nodes(canonical_etype[0])
            dst_num_nodes = graph[0].num_nodes(canonical_etype[-1])
            num_nodes_sum = src_num_nodes + dst_num_nodes
            num_edges = graph[0].num_edges(canonical_etype)
            std = nn.init.calculate_gain('relu')
            if num_nodes_sum > 0:
                std *= sqrt(2.0 / num_nodes_sum)
            edge_masks[canonical_etype] = nn.Parameter(
                torch.randn(num_edges, device=device) * std)


        return feat_masks, edge_masks
        # return feat_masks

    def _loss_regularize(self, loss, feat_masks, edge_masks):
        eps = 1e-15

        for feat_mask in feat_masks.values():
            feat_mask = feat_mask.sigmoid()
            # Feature mask sparsity regularization
            loss = loss + self.beta1 * torch.mean(feat_mask)
            # Feature mask entropy regularization
            ent = - feat_mask * torch.log(feat_mask + eps) - \
                  (1 - feat_mask) * torch.log(1 - feat_mask + eps)
            loss = loss + self.beta2 * ent.mean()

        for edge_mask in edge_masks.values():
            edge_mask = edge_mask.sigmoid()
            # Edge mask sparsity regularization
            loss = loss + self.alpha1 * torch.sum(edge_mask)
            # Edge mask entropy regularization
            ent = - edge_mask * torch.log(edge_mask + eps) - \
                (1 - edge_mask) * torch.log(1 - edge_mask + eps)
            loss = loss + self.alpha2 * ent.mean()


        return loss

    def explain_graph(self, graph, feat, **kwargs):
        self.model = self.model.to(device)
        # self.model.eval()
        # self.model.train()

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph, feat, **kwargs).squeeze()
            pred_label = logits.argmax(dim=-1)
            # pred_label = logits


        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description('Explain graph')


        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type, node_feat in feat.items():
                h[node_type] = node_feat * feat_mask[node_type].sigmoid()
            eweight = {}
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                eweight[canonical_etype] = canonical_etype_mask.sigmoid()


            logits = self.model(graph, h, **kwargs).squeeze()

            log_probs = logits.log_softmax(dim=-1)
            # log_probs = logits
            # print(log_probs)
            # print(pred_label[0])
            # loss = 0
            # loss = -[loss + log_probs[idx, pred_label[idx]] for idx in range(log_probs.size(0))]
            # print(loss)

            loss = F.nll_loss(
                input=log_probs,
                target=pred_label.long(),
            )

            loss = self._loss_regularize(loss, feat_mask, edge_mask)

            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = feat_mask[node_type].detach().sigmoid().squeeze()

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = edge_mask[canonical_etype].detach().sigmoid()

        return feat_mask, edge_mask



