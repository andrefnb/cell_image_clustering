#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:59:35 2019

@author: carolinagoldstein
"""

import tp2_aux
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import pairwise_distances
from numpy import genfromtxt
import json
import random
random.seed(1)

def standardize(data):
    means = np.mean(data,axis=0)
    stdevs = np.std(data,axis=0)
    data_std = (data-means)/stdevs
    return data_std

def plot_clusters(X,y_pred,title=''):
    """Plotting function; y_pred is an integer array with labels"""    
    plt.figure()
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.axis('equal')
    plt.show()

def get_count(labels,prediction):
    labelled,prediction_labelled = labelled_data(labels,prediction)
    labels_ = pairwise_distances(labelled.reshape(-1, 1))
    binary_labels = (abs(labels_)>0).astype(int)
    predictions_ = pairwise_distances(prediction_labelled.reshape(-1, 1))
    binary_predictions = (abs(predictions_)>0).astype(int)    
    TP=0
    FP=0
    TN=0
    FN=0
    length = len(labelled)
    for i in range(length):
        for j in range(length):
            if j<i:
                if binary_labels[i,j]==binary_predictions[i,j]==0:
                    TP+=1
                elif binary_labels[i,j]==binary_predictions[i,j]==1:
                    TN+=1
                elif binary_labels[i,j]==0 and binary_predictions[i,j]==1:
                    FN+=1
                else:
                    FP+=1                    
    return TP,FP,TN,FN

def rand_index(labelled,prediction_labelled):
    TP,FP,TN,FN = get_count(labelled,prediction_labelled)
    RI = (TP+TN)/(TP+FP+FN+TN)
    return RI

def labelled_data(labels,prediction):
    labelled_and_prediction_total = np.concatenate((labels[:,None],prediction[:,None]), axis=1)
    labelled_and_prediction = labelled_and_prediction_total[labelled_and_prediction_total[:,0]!=0]
    prediction_labelled = labelled_and_prediction[:,1]
    labelled = labelled_and_prediction[:,0]
    return prediction_labelled,labelled

def get_measures(labels, prediction):
    TP,FP,TN,FN = get_count(labels,prediction)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2 * ((precision*recall)/(precision+recall))
    return recall, precision, f1        

def plot_measures(alg_name, argument_name, alg_range, scores):
    rand_scores, adjusted_rand_scores, silhouette_scores, f1_scores, recall_scores, precision_scores = scores
    # Plot measure and indexes graph
    plt.plot(alg_range, rand_scores)
    plt.plot(alg_range, adjusted_rand_scores)
    plt.plot(alg_range, silhouette_scores)
    plt.plot(alg_range, f1_scores)
    plt.plot(alg_range, recall_scores)
    plt.plot(alg_range, precision_scores)
    plt.legend(['Rand', 'Adjusted Rand', 'Silhouette', 'F1', 'Recall', 'Precision'], loc = 'best', prop={'size': 7})
    plt.xlabel(argument_name)
    plt.ylabel("Scores")
    plt.savefig(f'{alg_name}_indicators.png')
    plt.show()
    plt.close()
    
def get_metrics(data_std, labels, method_range, method):
    rand_scores = []
    adjusted_rand_scores = []
    silhouette_scores = [] 
    f1_scores = []
    recall_scores = []
    precision_scores = []
        
    for i in method_range:
        if method == "KMeans":
            model = KMeans(n_clusters=i)            
        elif method == "GMM":
            model = GaussianMixture(n_components=i)
        else:
            model = DBSCAN(eps=i)
        prediction = model.fit_predict(data_std)    
        prediction_labelled,labelled = labelled_data(labels,prediction)
        
        try:
            silhouette_scores.append(silhouette_score(data_std,prediction))  
        except:
            silhouette_scores.append(0)

        adjusted_rand_scores.append(adjusted_rand_score(labelled,prediction_labelled))
        precision, recall, f1 = get_measures(labels, prediction)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        rand_scores.append(rand_index(labelled,prediction_labelled))
        
    return rand_scores, adjusted_rand_scores, silhouette_scores, f1_scores, recall_scores, precision_scores

# Import and standardize data (selected features)
data = genfromtxt('selected_features.csv', delimiter=',')
labels = genfromtxt('labels.txt', delimiter=',')[:,1]
data_std = standardize(data)

# Run Kmeans clustering
kmeans_range = range(2,13)
rand_scores, adjusted_rand_scores, silhouette_scores, f1_scores, recall_scores, precision_scores = get_metrics(data_std, labels, kmeans_range, "KMeans")
plot_measures("Kmeans", "K", kmeans_range, [rand_scores, adjusted_rand_scores, silhouette_scores, f1_scores, recall_scores, precision_scores])

# Run DBSCAN clustering
dbscan_range = np.linspace(0.1,1.1,21)
rand_scores_DB, adjusted_rand_scores_DB, silhouette_scores_DB, f1_scores_DB, recall_scores_DB, precision_scores_DB = get_metrics(data_std, labels, dbscan_range, "DBSCAN")
plot_measures("DBSCAN", "eps", dbscan_range, [rand_scores_DB, adjusted_rand_scores_DB, silhouette_scores_DB, f1_scores_DB, recall_scores_DB, precision_scores_DB])

# Best eps
# choosing best eps according to precision_score
scores_eps = pd.DataFrame(list(zip(dbscan_range, precision_scores_DB)), columns =['eps', 'score'])
best_eps = scores_eps.nlargest(3,['score'])['eps']
# Fit and predict
#model = DBSCAN(eps=best_eps[1])
#prediction = model.fit_predict(data_std)    
      

# Plots clusters - compare best k's and eps's
#plot_clusters(data_std, prediction)
print("KMEANS")
print(f'Adjusted rand scores:{adjusted_rand_scores}')
print(f'Rand scores:{rand_scores}')
print(f'Silhouette scores:{silhouette_scores}')
print(f'F1 scores:{f1_scores}')
print(f'Recall:{recall_scores}')
print(f'Precision:{precision_scores}')

print("DBSCAN")
print(f'Adjusted rand scores:{adjusted_rand_scores_DB}')
print(f'Rand scores:{rand_scores_DB}')
print(f'Silhouette scores:{silhouette_scores_DB}')
print(f'F1 scores:{f1_scores_DB}')
print(f'Recall:{recall_scores_DB}')
print(f'Precision:{precision_scores_DB}')

#DBSCAN - choosing the best eps according to article (Elbow Method)
n_neighbors = 5
y = np.zeros(np.size(data_std, axis=0))
neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
nbrs = neigh.fit(data_std, y)
distances, indices = nbrs.kneighbors(data_std)

distances = distances[:,n_neighbors-1]
distances = np.sort(distances, axis=0)
distances = distances[::-1]

idx = range(0,len(distances))
plt.scatter(idx, distances, marker='.')
noise_threshold = 40
# Noise threshold line
plt.axvline(x=noise_threshold, color = "orange")
# Choosing index for best distance value
best_distance = distances[noise_threshold]
print(f"Best Distance:{best_distance}")
# DBSCAN with best eps chosen with the Elbow Method
dbscan_elbow = DBSCAN(eps=best_distance).fit_predict(data_std)
plot_clusters(data_std, dbscan_elbow, title=f"DBSCAN with {round(best_distance,2)} as EPS")
ids = genfromtxt('labels.txt', delimiter=',')[:,0]
tp2_aux.report_clusters(ids, dbscan_elbow, 'cluster_dbscan_elbow.html')



ids = genfromtxt('labels.txt', delimiter=',')[:,0]
#tp2_aux.report_clusters(ids, labels, 'cluster_original.html')
#tp2_aux.report_clusters(ids, prediction, 'cluster_eps_0.6.html')


# Question  6
# K-Means k=2,3,4,5,9
for i in [2,3,4,5,9,11]:
    model = KMeans(n_clusters=i)  
    prediction = model.fit_predict(data_std)  
    tp2_aux.report_clusters(ids, prediction, 'cluster_km_{}.html'.format(i))
    
# Epsilon eps=0.45,0.5,0.55,0.6
for i in [0.4,0.45,0.5,0.55,0.6]:
    model = DBSCAN(eps=i) 
    prediction = model.fit_predict(data_std)  
    tp2_aux.report_clusters(ids, prediction, 'cluster_eps_{}.html'.format(i))    
    
# Epsilon eps=0.45,0.5,0.55,0.6
for i in [0.4,0.45,0.5,0.55,0.6]:
    model = DBSCAN(eps=i) 
    prediction = model.fit_predict(data_std)  
    tp2_aux.report_clusters(ids, prediction, 'cluster_eps_{}.html'.format(i))    
 
# GMM Algorithm
# Run GMM clustering
gmm_range = range(2,13)
rand_scores_GMM, adjusted_rand_scores_GMM, silhouette_scores_GMM, f1_scores_GMM, recall_scores_GMM, precision_scores_GMM = get_metrics(data_std, labels, gmm_range, "GMM")
plot_measures("GMM", "number of components", gmm_range, [rand_scores_GMM, adjusted_rand_scores_GMM, silhouette_scores_GMM, f1_scores_GMM, recall_scores_GMM, precision_scores_GMM])

# GMM n_components = 2,3,4,5,9,11
for i in [2,3,4,5,6,9,11]:
    model = GaussianMixture(n_components=i)
    prediction = model.fit_predict(data_std)  
    tp2_aux.report_clusters(ids, prediction, 'cluster_gmm_{}.html'.format(i))

    
    

    
# Bisecting KMeans
class Bisected_Branch:
    
    def __init__(self, data, parent, cluster_number):
        self.data = data
        self.children = []
        self.parent = parent
        if parent != None:
            self.history = parent.history.copy()
            self.history.append(cluster_number)
        else:
            self.history = []
            
    def isLeaf(self):
        return len(self.data) <= 1

    def divide(self):
        model = KMeans(n_clusters=2)
        prediction = model.fit_predict(self.data['row'].tolist())
        df = pd.DataFrame(list(zip(self.data['row_number'], self.data['row'], prediction)), columns =['row_number', 'row', 'class'])
        cluster_0 = df.loc[df['class'] == 0]
        cluster_1 = df.loc[df['class'] == 1]
        
        branch_0 = Bisected_Branch(cluster_0, self, 0)
        branch_1 = Bisected_Branch(cluster_1, self, 1)
        
        self.children = [branch_0, branch_1]
        return self.children
    
    def repr(self):
        return f"data length: {len(self.data)}, number children: {len(self.children)}"
        
      
def bisecting_kmeans_alg(data, depth = 100000000):
    root = Bisected_Branch(data, None, None)
    it = 0
    clusters = []
    branch = root
    clusters.append(branch)
    leafs = []
    outs = []
    while(len(clusters)>0 and it < depth and len(outs) < 5):
        it += 1
        
        # divide branch
        newer_branches = branch.divide()
        
        if (len(newer_branches[0].data) == 0 or len(newer_branches[1].data) == 0):
            outs.append(True)
        
        if newer_branches[0].isLeaf(): leafs.append(newer_branches[0])
        else: clusters.append(newer_branches[0])
        
        if newer_branches[1].isLeaf(): leafs.append(newer_branches[1])
        else: clusters.append(newer_branches[1])
        
        # take branch out of clusters
        for branch_index in range(len(clusters)):
            if (branch == clusters[branch_index]):
                del clusters[branch_index]
                break
        
        new_clusters = sorted(clusters, key=lambda x: len(getattr(x, 'data')), reverse = True)
        bigger_branch = new_clusters[0]
            
        # update branch
        branch = bigger_branch
        
    return clusters + leafs

def bisecting_kmeans(data, depth = 5):

    df = pd.DataFrame(list(zip(range(len(data)), data)), columns =['row_number', 'row'])
    clusters = bisecting_kmeans_alg(df, depth)
    
    datas = []
    for cluster in clusters:
        datas.append(pd.DataFrame({"row_number": cluster.data['row_number'], "row": cluster.data['row'], "history": str(cluster.history)}, columns =['row_number', 'row', 'history']))
        
    result = pd.concat(datas)
        
    results_ordered = result.sort_values(by=['row_number'])
    indexes = []
    list_labels = []
    for i, row in results_ordered.iterrows():
        indexes.append(row["row_number"])
        list_labels.append(json.loads(row["history"]))
        
    tp2_aux.report_clusters_hierarchical(indexes,list_labels,f"cluster_bisectk_{depth}.html")

# Iterate for Bissecting Kmeans
depth_range = range(2,16)
for i in depth_range:
    bisecting_kmeans(data_std, depth = i)
