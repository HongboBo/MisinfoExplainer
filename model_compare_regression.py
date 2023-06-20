from data import load_mumin_graph
from typing import Dict
import torch
import dgl
from models.graphsage import HeteroGraphSAGE
from models.gcn_model import HeteroGraphGCN
from models.gat_model import HeteroGraphGAT
import torch.utils.data as D
from dgl.dataloading import NeighborSampler
from dgl.dataloading import NodeDataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, R2Score
from models.HeteroGNNExplainer_mse import HeteroGNNExplainer



def train_graph_model(task: str,
                      size: str,
                      num_epochs: int,
                      random_split: bool = False,
                      gnn_model: str = 'sage',
                      **_) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(4242)
    dgl.seed(4242)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_mumin_graph(size=size)
    graph = dataset.to_dgl()


    '''misinformation label is 0, remove label 1'''
    nid = torch.nonzero(graph.nodes['tweet'].data['label'])
    graph.remove_nodes(nid.squeeze().int(), ntype='tweet')


    tweet_feat = graph.nodes['tweet'].data['feat'][:, -3:]
    sum_feat = tweet_feat.sum(dim=1)
    # trend = sum_feat/torch.mean(sum_feat)
    # trend = F.normalize(trend, p=2.0, dim=0)
    trend = torch.log(sum_feat + 1)


    graph.nodes['tweet'].data['trend'] = trend
    graph.nodes['tweet'].data['feat'] = graph.nodes['tweet'].data['feat'][:, :-3]
    graph.nodes['reply'].data['feat'] = graph.nodes['reply'].data['feat'][:, :-3]


    train_mask = graph.nodes[task].data['train_mask'].bool()
    val_mask = graph.nodes[task].data['val_mask'].bool()
    test_mask = graph.nodes[task].data['test_mask'].bool()



    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1] for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], dims[rel[2]]) for rel in graph.canonical_etypes}


    if gnn_model == 'sage':
        model = HeteroGraphSAGE(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=512,
                            feat_dict=feat_dict,
                            task=task)

    elif gnn_model == 'gcn':
        model = HeteroGraphGCN(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=512,
                            feat_dict=feat_dict,
                            task=task)
    elif gnn_model == 'gat':
        model = HeteroGraphGAT(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=512,
                            feat_dict=feat_dict,
                            task=task)
    model.to(device)
    # model.load_state_dict(torch.load(gnn_model+size+'.pth'))
    model.train()

    node_enum = torch.arange(graph.num_nodes(task))
    if random_split:
        torch_gen = torch.Generator().manual_seed(4242)

        num_train = int(0.8 * graph.num_nodes(task))
        num_val = int(0.1 * graph.num_nodes(task))
        num_test = graph.num_nodes(task) - (num_train + num_val)
        nums = [num_train, num_val, num_test]


        train_nids, val_nids, test_nids = D.random_split(dataset=node_enum.int(), lengths=nums, generator=torch_gen)


        train_nids = {task: train_nids}
        val_nids = {task: val_nids}
        test_nids = {task: test_nids}

    else:
        train_nids = {task: node_enum[train_mask].int()}
        val_nids = {task: node_enum[val_mask].int()}
        test_nids = {task: node_enum[test_mask].int()}


    sampler = NeighborSampler([100, 100], replace=False)

    train_dataloader = NodeDataLoader(graph=graph,
                                      indices=train_nids,
                                      graph_sampler=sampler,
                                      batch_size=32,
                                      shuffle=True,
                                      drop_last=True)
    val_dataloader = NodeDataLoader(graph=graph,
                                    indices=val_nids,
                                    graph_sampler=sampler,
                                    batch_size=1000000,
                                    shuffle=False,
                                    drop_last=False)
    test_dataloader = NodeDataLoader(graph=graph,
                                     indices=test_nids,
                                     graph_sampler=sampler,
                                     batch_size=1000000,
                                     shuffle=False,
                                     drop_last=False)


    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    scheduler = LinearLR(optimizer=opt, start_factor=1., end_factor=1e-7 / 3e-4, total_iters=100)



    epoch_pbar = tqdm(range(num_epochs), desc='Training')

    for epoch in epoch_pbar:
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for _, _, blocks in train_dataloader:

            opt.zero_grad()
            blocks = [block.to(device) for block in blocks]


            input_feats = {n: feat.float() for n, feat in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['trend'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats).squeeze()

            # Compute loss
            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )




            loss.backward()
            opt.step()
            train_loss += float(loss)

        train_loss /= len(train_dataloader)







        model.eval()
        for _, _, blocks in val_dataloader:
            with torch.no_grad():
                blocks = [block.to(device) for block in blocks]

                input_feats = {n: f.float() for n, f in blocks[0].srcdata['feat'].items()}
                output_labels = blocks[-1].dstdata['trend'][task].to(device)
                logits = model(blocks, input_feats).squeeze()


                loss = F.mse_loss(
                    input=logits,
                    target=output_labels.float(),
                )

                val_loss += float(loss)

        val_loss /= len(val_dataloader)


        # Update progress bar description
        desc = (f'Training - '
                f'loss {train_loss:.3f} - '
                f'val_loss {val_loss:.3f} - '
                )
        epoch_pbar.set_description(desc)
        scheduler.step()

    # Close progress bar
    epoch_pbar.close()
    val_loss = 0.0
    test_loss = 0.0

    # Final evaluation on the validation set
    model.eval()
    for _, _, blocks in tqdm(val_dataloader, desc='Evaluating'):
        with torch.no_grad():
            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            # output_labels = blocks[-1].dstdata['label'][task].to(device)

            output_labels = blocks[-1].dstdata['trend'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats).squeeze()

            # Compute validation loss
            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )
            # Store the validation loss
            val_loss += float(loss)

    # Divide the validation loss by the number of batches
    val_loss /= len(val_dataloader)

    # Final evaluation on the test set

    MAPE = MeanAbsolutePercentageError().to(device)
    MSE = MeanSquaredError().to(device)
    r2score = R2Score().to(device)


    model.eval()
    for _, _, blocks in tqdm(test_dataloader, desc='Evaluating'):
        with torch.no_grad():
            blocks = [block.to(device) for block in blocks]

            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            # output_labels = blocks[-1].dstdata['label'][task].to(device)
            output_labels = blocks[-1].dstdata['trend'][task].to(device)

            logits = model(blocks, input_feats).squeeze()
            torch.save(logits, 'test.pth')


            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )

            test_loss += float(loss)

            MAPE(logits, output_labels)
            MSE(logits, output_labels)
            r2score(logits, output_labels)



    test_loss /= len(test_dataloader)

    test_mape = MAPE.compute().item()
    test_mse = MSE.compute().item()
    r2 = r2score.compute().item()


    results = {
        'train': {
            'loss': train_loss,
        },
        'val': {
            'loss': val_loss,
        },
        'test': {
            'loss': test_loss,
            'test_mape': test_mape,
            'test_mse': test_mse,
            'r2': r2,
        }
    }

    torch.save(model.state_dict(), gnn_model+'_'+size+'.pth')

    return results




results = train_graph_model(task='tweet', size='small', num_epochs=100, random_split=False, gnn_model='sage')
print(results)
