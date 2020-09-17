#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:44:54 2019

@author: carolinagoldstein
"""

import tp2_aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.manifold import Isomap
import numpy as np
import random

random.seed(1)

data = tp2_aux.images_as_matrix()

#PCA
pca = PCA(n_components=6)
pca.fit(data)
t_pca_data = pca.transform(data)

#TSNE
tsne = TSNE(n_components=6,method='exact')
t_tsne_data = tsne.fit_transform(data)

#Isometric
isomap = Isomap(n_components=6)
isomap.fit(data)
t_isomap_data = isomap.transform(data)

data_18 = np.concatenate((t_pca_data,t_tsne_data,t_isomap_data),axis=1)
np.savetxt('features.csv', data_18, delimiter=',', fmt='%d')