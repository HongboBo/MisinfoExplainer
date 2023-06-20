import numpy as np

path = '../../../../../media/secure/hongbo/DeepInf_data/weibo/label.npy'
labels = np.load(path)
c = np.bincount(labels)
print(c)