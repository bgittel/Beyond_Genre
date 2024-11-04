# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:56:05 2024

@author: KeliDu
"""
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist
import seaborn as sns
from matplotlib import pyplot as plt

corpus_path = r'F:\Kritik_projekt\3_Korpora_BBAW_normalized\all_renamed'
filenames = sorted([os.path.join(corpus_path, fn) for fn in os.listdir(corpus_path)])

groups = []
for name in filenames:
    group_name = os.path.basename(name).split('_')[0]
    if group_name == 'zeitkriCategorical':
        groups.append('critique of the present')
    if group_name == 'gesellschaftskriCategorical':
        groups.append('society critique')
    if group_name == 'sozialkriCategorical': 
        groups.append('social critique')
    if group_name == 'kulturkriCategorical':
        groups.append('cultural critique')
    if group_name == 'modernekriCategorical':
        groups.append('modernity critique')
    if group_name == 'zivilisationskriCategorical':
        groups.append('civilisation critique')
    if group_name == 'fortschrittskriCategorical':
        groups.append('progress critique')
    if group_name == 'dekadenzCategorical':
        groups.append('decadence literature')
    if group_name == 'heimatkunstCategorical':
        groups.append('regional heritage art')
    if group_name == 'worldview':
        groups.append('worldview literature')
    if group_name == 'conservative':
        groups.append('conservative revolution literature')
    if group_name == 'ELTEC':
        groups.append('ELTEC')
    if group_name =='multi':
        groups.append('multiple categories')
    if group_name == 'adventure':
        groups.append('adventure novels')

groups_count = Counter(groups)

vectorizer = CountVectorizer(input="filename")
dtm = vectorizer.fit_transform(filenames)
vocab = vectorizer.get_feature_names_out()
dtm_df = pd.DataFrame(dtm.toarray(), columns=vocab)

mfw_no = 2000
mfw_list = dtm_df.sum().sort_values(ascending=False)
mfs_vocab = mfw_list.index.tolist()[mfw_no:]
mfw_df = dtm_df[mfs_vocab]
mfw_df['category'] = groups

dist_dict = []
for group in groups_count.keys():
    df_part = mfw_df[mfw_df['category'] == group]
    df_part = df_part.drop('category', axis=1)
    distance = pdist(df_part, metric='cosine')
    for i in distance:
        dist_dict.append([group, i])
        
dist_df = pd.DataFrame(dist_dict, columns=['category', 'cosine_distance'])    
dist_df = dist_df.astype({"category": str})  

all_critiques = dist_df[~dist_df['category'].isin(['ELTEC', 'adventure novels'])]
all_critiques_mean = all_critiques['cosine_distance'].mean()
dist_df.loc[len(dist_df)] = ['Modernity Critique Corpus']  + [all_critiques_mean]

sorter = ['ELTEC', "Modernity Critique Corpus", 'adventure novels', 'civilisation critique', 'conservative revolution literature', 'critique of the present', 'cultural critique', 'decadence literature', 'modernity critique', 'multiple categories', 'progress critique', 'regional heritage art', 'social critique', 'society critique', 'worldview literature'] 
sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (12,10))
g = sns.pointplot(x='cosine_distance', y='category', data=dist_df, order = sorter, palette={'b'})
plt.show()


    