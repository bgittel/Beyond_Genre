# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:46:37 2024

@author: KeliDu
"""

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import os
 

#########################################################################################################################
#radar chart

cols = ['doc_names','Face', 'Neighborhood', 'Philosophy', 'Present', 'Seafaring', 'War ', 'Nature', 'Character', 'Sense of Time', 'Work', 'Movement', 'Religion', 'Grace', 'Night', 'Woman', 'Memory', 'Travel', 'House', 'Castle', 'Happiness', 'Village', 'War', 'Society', 'Locomotion', 'Animal', 'Body', 'Condition', 'Feeling', 'Moment', 'School']
    
df = pd.read_csv(r'F:\Kritik_projekt\results\NMF_doctopic_book_30topics_417.csv', sep='\t', names=cols)

group = []
for name in df['doc_names']:
    group_name = name.split('_')[0]
    if group_name == 'zeitkriCategorical':
        group.append('critique of the present')
    if group_name == 'gesellschaftskriCategorical':
        group.append('society critique')
    if group_name == 'sozialkriCategorical': 
        group.append('social critique')
    if group_name == 'kulturkriCategorical':
        group.append('cultural critique')
    if group_name == 'modernekriCategorical':
        group.append('modernity critique')
    if group_name == 'zivilisationskriCategorical':
        group.append('civilisation critique')
    if group_name == 'fortschrittskriCategorical':
        group.append('progress critique')
    if group_name == 'dekadenzCategorical':
        group.append('decadence literature')
    if group_name == 'heimatkunstCategorical':
        group.append('regional heritage art')
    if group_name == 'worldview':
        group.append('worldview literature')
    if group_name == 'conservative':
        group.append('conservative revolution literature')
    if group_name == 'ELTEC':
        group.append('ELTEC')
    if group_name =='multi':
        group.append('multiple categories')

df['group'] = group
 
df_avg = df[df.columns[~df.columns.isin(['doc_names'])]].groupby('group').mean()
 
rest_mean = df_avg[~df_avg.index.isin(['ELTEC'])].mean()
rest_mean.name = 'modernity critique corpus'
df_avg = df_avg.append(rest_mean)
df_avg = df_avg.drop(index=['progress critique'])

os.chdir (r'F:\Kritik_projekt\results\radar_plots_30topics_417')
 
# number of variable
categories = cols[1:]
N = len(categories)
 
row = 1
while row < len(df_avg):

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    f = plt.figure(figsize=(16,10))
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=15)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="black", size=14)
    plt.ylim(0,max(df_avg.max())*1.01)
    # Plot data
    
    values=df_avg.iloc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1.5, linestyle='solid', label="ELTEC")
    ax.fill(angles, values, 'b', alpha=0.1)
     
    # Ind2
    values=df_avg.iloc[row].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=df_avg.index[row])
    ax.fill(angles, values, 'r', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    #plt.title(df_avg.index[row] + '_vs_ELTEC')
    
    f.savefig(df_avg.index[row] + '_vs_ELTEC.png')
    row += 1
    
