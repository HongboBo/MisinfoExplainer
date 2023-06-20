from data import load_mumin_graph
from typing import Dict
import torch
import dgl
from models.graphsage import HeteroGraphSAGE
import torch.utils.data as D
from dgl.dataloading import NeighborSampler
from dgl.dataloading import NodeDataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm
import torch.nn.functional as F
from models.HeteroCaptum import HeteroGNNExplainer



def train_graph_model(task: str,
                      size: str,
                      num_epochs: int,
                      random_split: bool = False,
                      **_) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(4242)
    dgl.seed(4242)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_mumin_graph(size=size)
    graph = dataset.to_dgl()

    # for item in graph.nodes['tweet']:
    #     print(item)
    #     exit()
    #     graph.remove_nodes()
    # if graph.nodes['tweet'].data['label'] == 0:
    #     print()

    '''misinformation label is 0, remove label 1'''
    nid = torch.nonzero(graph.nodes['tweet'].data['label'])
    graph.remove_nodes(nid.squeeze().int(), ntype='tweet')

    tweet_feat = graph.nodes['tweet'].data['feat'][:, -3:]
    sum_feat = tweet_feat.sum(dim=1)
    trend = sum_feat/torch.mean(sum_feat)

    graph.nodes['tweet'].data['trend'] = trend
    graph.nodes['tweet'].data['feat'] = graph.nodes['tweet'].data['feat'][:, :-3]
    graph.nodes['reply'].data['feat'] = graph.nodes['reply'].data['feat'][:, :-3]


    train_mask = graph.nodes[task].data['train_mask'].bool()
    val_mask = graph.nodes[task].data['val_mask'].bool()
    test_mask = graph.nodes[task].data['test_mask'].bool()

    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1] for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], dims[rel[2]]) for rel in graph.canonical_etypes}

    model = HeteroGraphSAGE(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=512,
                            feat_dict=feat_dict,
                            task=task)
    model.to(device)
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
                                      drop_last=False)
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

    # pos_weight_tensor = torch.tensor(20.).to(device)
    # class_weights = [1., 20.]
    # pos_weight_tensor = torch.tensor(class_weights).to(device)

    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    scheduler = LinearLR(optimizer=opt, start_factor=1., end_factor=1e-7 / 3e-4, total_iters=100)

    # scorer = tm.classification.F1Score(task="multiclass", num_classes=2, average='none').to(device)

    epoch_pbar = tqdm(range(num_epochs), desc='Training')

    for epoch in epoch_pbar:
        train_loss = 0.0
        val_loss = 0.0

        # scorer.reset()

        model.train()
        for _, _, blocks in train_dataloader:

            opt.zero_grad()
            blocks = [block.to(device) for block in blocks]


            input_feats = {n: feat.float() for n, feat in blocks[0].srcdata['feat'].items()}
            # output_labels = blocks[-1].dstdata['label'][task].to(device)
            output_labels = blocks[-1].dstdata['trend'][task].to(device)

            edge_masks = {}
            for c_etype in blocks[0].canonical_etypes:
                edge_masks[c_etype] = torch.ones(blocks[0].num_edges(c_etype)).to(device)
                edge_masks[c_etype].requires_grad = True

            # Forward propagation
            logits = model(blocks, input_feats, eweight= edge_masks).squeeze()

            # Compute loss
            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )

            # scorer(logits.argmax(axis=-1), output_labels)
            #
            loss.backward()
            opt.step()

            train_loss += float(loss)



        train_loss /= len(train_dataloader)

        # train_f1s = scorer.compute()
        # train_misinformation_f1 = train_f1s[0].item()
        # train_factual_f1 = train_f1s[1].item()

        # scorer.reset()



        model.eval()
        for _, _, blocks in val_dataloader:
            with torch.no_grad():
                blocks = [block.to(device) for block in blocks]

                input_feats = {n: f.float() for n, f in blocks[0].srcdata['feat'].items()}
                # output_labels = blocks[-1].dstdata['label'][task].to(device)
                output_labels = blocks[-1].dstdata['trend'][task].to(device)

                edge_masks = {}
                for c_etype in blocks[0].canonical_etypes:
                    edge_masks[c_etype] = torch.ones(blocks[0].num_edges(c_etype)).to(device)
                    edge_masks[c_etype].requires_grad = True

                # Forward propagation
                logits = model(blocks, input_feats, eweight=edge_masks).squeeze()


                loss = F.mse_loss(
                    input=logits,
                    target=output_labels.float(),
                )

                # scorer(logits.argmax(axis=-1), output_labels)
                val_loss += float(loss)

        val_loss /= len(val_dataloader)
        #
        # val_f1s = scorer.compute()
        # val_misinformation_f1 = val_f1s[0].item()
        # val_factual_f1 = val_f1s[1].item()

        # Update progress bar description
        desc = (f'Training - '
                f'loss {train_loss:.3f} - '
                # f'factual_f1 {train_factual_f1:.3f} - '
                # f'misinfo_f1 {train_misinformation_f1:.3f} - '
                f'val_loss {val_loss:.3f} - '
                # f'val_factual_f1 {val_factual_f1:.3f} - '
                # f'val_misinfo_f1 {val_misinformation_f1:.3f}'
                )
        epoch_pbar.set_description(desc)
        scheduler.step()

    # Close progress bar
    epoch_pbar.close()
    val_loss = 0.0
    test_loss = 0.0

    # Reset metrics
    # scorer.reset()






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
            edge_masks = {}
            for c_etype in blocks[0].canonical_etypes:
                edge_masks[c_etype] = torch.ones(blocks[0].num_edges(c_etype)).to(device)
                edge_masks[c_etype].requires_grad = True

            # Forward propagation
            logits = model(blocks, input_feats, eweight=edge_masks).squeeze()

            # Compute validation loss
            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )

            # Compute validation metrics
            # scorer(logits.argmax(axis=-1), output_labels)

            # Store the validation loss
            val_loss += float(loss)

    # Divide the validation loss by the number of batches
    val_loss /= len(val_dataloader)

    # Compute the validation metrics
    # val_f1s = scorer.compute()
    # val_misinformation_f1 = val_f1s[0].item()
    # val_factual_f1 = val_f1s[1].item()

    # Reset the metrics
    # scorer.reset()

    # Final evaluation on the test set
    model.eval()
    for _, _, blocks in tqdm(test_dataloader, desc='Evaluating'):
        with torch.no_grad():
            blocks = [block.to(device) for block in blocks]

            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            # output_labels = blocks[-1].dstdata['label'][task].to(device)
            output_labels = blocks[-1].dstdata['trend'][task].to(device)

            edge_masks = {}
            for c_etype in blocks[0].canonical_etypes:
                edge_masks[c_etype] = torch.ones(blocks[0].num_edges(c_etype)).to(device)
                edge_masks[c_etype].requires_grad = True

            # Forward propagation
            logits = model(blocks, input_feats, eweight=edge_masks).squeeze()


            loss = F.mse_loss(
                input=logits,
                target=output_labels.float(),
            )


            # scorer(logits.argmax(axis=-1), output_labels)

            test_loss += float(loss)



    test_loss /= len(test_dataloader)

    # test_f1s = scorer.compute()
    # test_misinformation_f1 = test_f1s[0].item()
    # test_factual_f1 = test_f1s[1].item()

    results = {
        'train': {
            'loss': train_loss,
            # 'factual_f1': train_factual_f1,
            # 'misinformation_f1': train_misinformation_f1
        },
        'val': {
            'loss': val_loss,
            # 'factual_f1': val_factual_f1,
            # 'misinformation_f1': val_misinformation_f1
        },
        'test': {
            'loss': test_loss,
            # 'factual_f1': test_factual_f1,
            # 'misinformation_f1': test_misinformation_f1
        }
    }

    '''Explainer'''
    model.train()
    feat_masks = {}
    edge_masks = {}
    i = 0
    explainer = HeteroGNNExplainer(model, num=1).to(device)




    for _, _, blocks in test_dataloader:
        blocks = [block.to(device) for block in blocks]
        input_feats = {n: feat.float() for n, feat in blocks[0].srcdata['feat'].items()}
        output_labels = blocks[-1].dstdata['trend'][task].to(device)
        feat_mask, edge_mask = explainer.explain_graph(blocks, input_feats, output_labels)


        for node_type, feature in input_feats.items():
            if i == 0:
                feat_masks[node_type] = feat_mask[node_type]
            else:
                feat_masks[node_type] += feat_mask[node_type]
        for canonical_etype, canonical_etype_mask in edge_mask.items():
            if i == 0:
                edge_masks[canonical_etype] = edge_mask[canonical_etype]
            else:
                edge_masks[canonical_etype] += edge_mask[canonical_etype]
        i += 1


    importance = {}

    print(len(feat_masks['reply']))

    importance['tweet_text'] = torch.mean(feat_masks['tweet'][:768]/i)
    importance['tweet_lang'] = torch.mean(feat_masks['tweet'][768:810]/i)


    importance['claim_embedding'] = torch.mean(feat_masks['claim'][:768]/i)
    importance['claim_reviewer'] = torch.mean(feat_masks['claim'][768:]/i)

    importance['image'] = torch.mean(feat_masks['image']/i)
    importance['hashtag'] = torch.mean(feat_masks['hashtag']/i)
    importance['user'] = torch.mean(feat_masks['user']/i) #user has many features

    importance['reply_text'] = torch.mean(feat_masks['reply'][:768]/i)
    importance['reply_lang'] = torch.mean(feat_masks['reply'][768:829]/i)

    etypes_dict = [[('tweet', 'discusses', 'claim'), ('claim', 'discusses_inv', 'tweet')],
                   [('tweet', 'has_hashtag', 'hashtag'), ('hashtag', 'has_hashtag_inv', 'tweet')],
                   [('user', 'has_hashtag', 'hashtag'), ('hashtag', 'has_hashtag_inv', 'user')],
                   [('tweet', 'has_image', 'image'), ('image', 'has_image_inv', 'tweet')],
                   [('user', 'posted', 'reply'), ('reply', 'posted_inv', 'user')],
                   [('reply', 'quote_of', 'tweet'), ('tweet', 'quote_of_inv', 'reply')],
                   [('reply', 'reply_to', 'tweet'), ('tweet', 'reply_to_inv', 'reply')],
                   [('tweet', 'mentions', 'user'), ('user', 'mentions_inv', 'tweet')],
                   [('user', 'posted', 'tweet'), ('tweet', 'posted_inv', 'user')],
                   [('user', 'retweeted', 'tweet'), ('tweet', 'retweeted_inv', 'user')],
                   [('user', 'follows', 'user'), ('user', 'follows_inv', 'user')],
                   [('user', 'mentions', 'user'), ('user', 'mentions_inv', 'user')]]

    for etype_pair in etypes_dict:
        importance[etype_pair[0]] = (torch.mean(edge_masks[etype_pair[0]] / i) + torch.mean(
            edge_masks[etype_pair[1]] / i)) / 2

    # for canonical_etype in edge_mask:
    #     imp = torch.sum(edge_masks[canonical_etype]/i)
    #     if imp.isnan():
    #         importance[canonical_etype] = torch.tensor(0.)
    #     else:
    #         importance[canonical_etype] = imp


    sum_im = sum(importance.values())

    for item in importance.keys():
        importance[item] = importance[item]/sum_im

    print(importance)

    return results, importance

results, importance = train_graph_model(task='tweet', size='small', num_epochs=100, random_split=False)
file = 'log2.txt'
with open(file, 'a') as f:
    f.write(str(importance))
    f.write('\n')
print(results)
