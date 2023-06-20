from torch import nn
from math import sqrt
import torch
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroGNNExplainer(nn.Module):
    def __init__(self, model):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model

    def explain_graph(self, block, feat, label, **kwargs):
        self.model = self.model.to(device)
        self.model.zero_grad()




        pred_label = self.model(block, feat, **kwargs).squeeze()
        loss = F.mse_loss(
            input=pred_label,
            target=label.float(),
        )
        loss.backward()

        # print(self.model.conv2.mods)

        for item in self.model.conv2.mods.keys():
            grads = self.model.conv2.mods[item].fc_neigh.weight.grad
            gradients = torch.mean(grads, dim=0)
            CAM = torch.relu(gradients)
            print(CAM.shape)
            exit()
        grads = self.model.conv2.mods.weight.grad
        gradients = torch.mean(grads, dim=0)
        CAM = torch.relu(gradients)
        print(CAM.shape)
        exit()





        # CAM = (grad_values[None, :, None, None] * inputs).sum(dim=1)
        #
        #
        # node_grads = self.model.fc1.weight.grad[target_class_idx, :]
        # activations = self.model.fc1(node_feats)
        # node_grad_cam = torch.sum(node_grads * activations, dim=1)
        # node_grad_cam = F.relu(node_grad_cam)
        # edge_grads = self.model.fc2.weight.grad[:, target_class_idx]
        # activations = self.model.mean_nodes(graph, node_feats)
        # edge_grad_cam = torch.sum(edge_grads * activations, dim=1)
        # edge_grad_cam = F.relu(edge_grad_cam)
        # return node_grad_cam, edge_grad_cam






