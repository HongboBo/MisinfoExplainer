import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from data import load_mumin_graph
import torch


sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5)})

dataset = load_mumin_graph(size='small')
graph = dataset.to_dgl()
tweet_feat = graph.nodes['tweet'].data['feat'][:, -3:]
sum_feat = tweet_feat.sum(dim=1)
# trend = sum_feat/torch.mean(sum_feat)
trend = torch.log(sum_feat + 1)



plt.hist(trend.numpy(), bins=80, histtype="stepfilled", alpha=.8)
plt.savefig("dis.pdf", bbox_inches='tight')
plt.show()