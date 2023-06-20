import dgl
import dgl.nn as dglnn
import torch

import torch.nn.functional as F
from captum.attr import IntegratedGradients
from functools import partial

dataset = dgl.data.MUTAGDataset()

g = dataset[0]

# Some synthetic node features - replace it with your own.
for ntype in g.ntypes:
    g.nodes[ntype].data['x'] = torch.randn(g.num_nodes(ntype), 10)


# Your model...
class Model(torch.nn.Module):
    def __init__(self, etypes):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({etype: dglnn.SAGEConv(10, 10, 'mean') for _, etype, _ in etypes})
        self.w = dglnn.EdgeWeightNorm()
        self.l = torch.nn.Linear(10,1)
    def forward(self, blocks, h_dict, eweight = None):
        if eweight is None:
            h_dict = self.conv1(blocks[0], h_dict)
            output_sum = 0
        else:
            h_dict = self.conv1(blocks[0], h_dict, mod_kwargs={
            c_etype: {'edge_weight': self.w(blocks[0][c_etype], eweight[c_etype])} for c_etype in blocks[0].canonical_etypes
            })
            output_sum = sum(
                torch.sum(self.w(blocks[0][c_etype], eweight[c_etype])) for c_etype in blocks[0].canonical_etypes)
        return self.l(h_dict['SCHEMA']) + output_sum


m = Model(g.canonical_etypes)

# Minibatch sampling stuff...
sampler = dgl.dataloading.as_edge_prediction_sampler(
    dgl.dataloading.NeighborSampler([2, 2]), negative_sampler=dgl.dataloading.negative_sampler.Uniform(2))
eid = {g.canonical_etypes[0]: torch.arange(g.num_edges(g.canonical_etypes[0]))}
# Let's iterate over and explain one edge at a time.
dl = dgl.dataloading.DataLoader(g, eid, sampler, batch_size=1)


# Define a function that takes in a single tensor as the first argument and also returns a
# single tensor.

def forward_model(edge_masks, c_etype,  blocks, x_dict, eweight):
    # eweight = eweight.copy()
    # eweight[c_etype] = x
    eweight[c_etype] = edge_masks
    return m(blocks, x_dict, eweight=eweight).squeeze()


for input_nodes, pair_graph, neg_pair_graph, blocks in dl:
    input_dict = blocks[0].ndata['x']

    output = m(blocks, input_dict)

    loss = F.mse_loss(
        input=output,
        target=torch.randn(output.size()).float(),
    )
    loss.backward()


    edge_masks = {}
    for c_etype in blocks[0].canonical_etypes:
        edge_masks[c_etype] = torch.ones(blocks[0].num_edges(c_etype))
        edge_masks[c_etype].requires_grad = True

    for c_etype in blocks[0].canonical_etypes:
        # ig = IntegratedGradients(partial(forward_model,
        #                                  c_etype=c_etype,  blocks=blocks, x_dict=input_dict, eweight=edge_masks))
        # print(ig.attribute(edge_masks[c_etype], internal_batch_size=edge_masks[c_etype].size(0), n_steps=50))
        ig = IntegratedGradients(forward_model)
        print(ig.attribute(edge_masks[c_etype], additional_forward_args=(c_etype, blocks, input_dict, edge_masks),
                            internal_batch_size=edge_masks[c_etype].size(0), n_steps=50))
        break