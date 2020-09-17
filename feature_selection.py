#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:31:38 2019

@author: carolinagoldstein
"""

import  numpy as np
from numpy import genfromtxt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from mpl_toolkits.mplot3d import Axes3D 

def heatmap(data,name):
    df = pd.DataFrame(data)
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(abs(cor), annot=True, cmap=plt.cm.Reds)
    plt.savefig(name)
    plt.close()
    return cor

def standardize(data):
    means = np.mean(data,axis=0)
    stdevs = np.std(data,axis=0)
    data_std = (data-means)/stdevs
    return data_std

def plot_2features(X,y,best1,best2,file_name, total = False):
    plt.figure(figsize=(7,7))
    if total:
        plt.plot(X[y==1,best1], X[y==1,best2],'o', markersize=7, color='blue', alpha=0.5) 
        plt.plot(X[y==2,best1], X[y==2,best2],'o', markersize=7, color='red', alpha=0.5) 
        plt.plot(X[y==3,best1], X[y==3,best2],'o', markersize=7, color='green', alpha=0.5) 
        plt.plot(X[y==0,best1], X[y==0,best2],'o', markersize=7, color='gray', alpha=0.5) 
    else:
        plt.plot(X[y==1,best1], X[y==1,best2],'o', markersize=7, color='blue', alpha=0.5) 
        plt.plot(X[y==2,best1], X[y==2,best2],'o', markersize=7, color='red', alpha=0.5) 
        plt.plot(X[y==3,best1], X[y==3,best2],'o', markersize=7, color='green', alpha=0.5) 
    plt.gca().set_aspect('equal', adjustable='datalim',share=True)
    plt.gca().set_ybound(lower=-9, upper=9)
    plt.gca().set_xbound(lower=-9, upper=9)
    plt.savefig(file_name,dpi=200,bbox_inches='tight')
    plt.close()

# Import and standardize data    
data = genfromtxt('features.csv', delimiter=',')
labels = genfromtxt('labels.txt', delimiter=',')[:,1]
data_std = standardize(data)

# Use pearson correlation to filter out redundant features (18->13)
cor = heatmap(data_std,'corr_18.png')
high_correlations={}
for i in range(18):
    for j in range(18):
        if j>i:
            if abs(cor[i][j])>0.5:
                high_correlations[f'{i},{j}']=cor[i][j]

# Eliminate features that are highly correlated to others -> retain 13 features
data_std_red = np.delete(data_std,[0,3,13,14,16],1)
cor_2 = heatmap(data_std_red,'corr_13.png')

# ANOVA F-test with the labelled observations (n=81) to eliminate features that seem independent from class labels 
data_anova = np.concatenate((labels[:,None],data_std_red), axis=1)
data_anova_usable = data_anova[data_anova[:,0]!=0]
X = data_anova_usable[:,1:]
Y = data_anova_usable[:,0]

f_values, probs = f_classif(X, Y)

features_to_delete = []
for i in range(len(probs)):
    if probs[i]>0.01:
        features_to_delete.append(i)

# Eliminate features that seem independent from class labels at 1% significance level -> retain 4 features
data_selected = np.delete(data_std_red,[2, 3, 4, 5, 6, 7, 8, 11, 12],1)
# Plot 2 features that are less independent from class labels
plot_2features(X,Y,0,10,'best2features_labelled.jpg') # using the 81 labeled observations
plot_2features(data_std_red,labels,0,10,'best2features_complete.jpg',True) # using the whole dataset (563 observations)

# Plot 4 retained features pairwise to check if there are any dependencies not detected by correlation
plot_2features(data_selected,labels,0,1,'features_0_1.jpg',True)
plot_2features(data_selected,labels,0,2,'features_0_2.jpg',True)
plot_2features(data_selected,labels,0,3,'features_0_3.jpg',True)
plot_2features(data_selected,labels,1,2,'features_1_2.jpg',True)
plot_2features(data_selected,labels,1,3,'features_1_3.jpg',True)
plot_2features(data_selected,labels,2,3,'features_2_3.jpg',True)

# Eliminate feature that seems related to all other features -> retain 3 features
data_selected_final = np.delete(data_selected,[2],1)

# Save selected features in csv file (3 features)
np.savetxt('selected_features.csv', data_selected_final, delimiter=',')

# Plots 3D plot of the 3 selected features
fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(data_selected_final[:,0], data_selected_final[:,1], data_selected_final[:,2], s=10) 
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_zlabel('feature 3')
ax.set_xlim3d(-3,5)
ax.set_ylim3d(-3,5)
ax.set_zlim3d(-3,5)
plt.savefig('features.png',dpi=200,bbox_inches='tight')
plt.close()