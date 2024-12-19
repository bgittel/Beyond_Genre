# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 01:32:28 2024

@author: KeliDu
"""
import os
import pandas as pd
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import colorcet as cc

######################################################################################################################
#NMF topic modeling

segmentfolder = r'F:\Kritik_projekt\3_Korpora_BBAW_normalized\lemma_NA_segments'
segment_files = sorted([os.path.join(segmentfolder, fn) for fn in os.listdir(segmentfolder)]) 

stopwords = open(r'F:\Kritik_projekt\Stopwords_for_TM.txt', 'r', encoding='utf-8').read().split('\n')

vectorizer = CountVectorizer(input='filename', stop_words=stopwords)
dtm = vectorizer.fit_transform(segment_files)
vocab = np.array(vectorizer.get_feature_names_out())
tf_idf_m = TfidfTransformer(use_idf=False).fit_transform(dtm)   

num_topics = 30
num_top_words = 10

clf = decomposition.NMF(n_components=num_topics, random_state=1)
segment_topic = clf.fit_transform(tf_idf_m)

topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
    
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t])))
    
segment_topic = segment_topic / np.sum(segment_topic, axis=1, keepdims=True)

cols = []
n = 0
while n < num_topics:
    cols.append('topic_' + str(n))
    n+=1

segment_topic_df = pd.DataFrame(segment_topic, columns=cols)

segment_names = []
document_names = []
for file in segment_files:
    segment_names.append(os.path.basename(file)[:-4])
    document_names.append(os.path.basename(file)[:-8])
segment_topic_df['segment_names'] = segment_names
segment_topic_df['document_names'] = document_names

group = []
for name in segment_topic_df['segment_names']:
    group.append(name.split('_')[0])
segment_topic_df['group'] = group

segment_topic_df.to_csv(r'NMF_doctopic_segments_30topics.csv', sep='\t', index=False)

######################################################################################################################
#get mean topic document distribution from topic segment distribution

document_names = sorted(list(set(document_names)))

document_topic = []
doc_no = 0
while doc_no < len(document_names):
    segs_topic_df = segment_topic_df[segment_topic_df['document_names'] == document_names[doc_no]]
    segs_topic_df = segs_topic_df.drop(columns=['segment_names', 'document_names', 'group'])
    doc_topic = list(segs_topic_df.mean())
    document_topic.append(([document_names[doc_no]] + doc_topic))        
    doc_no += 1
    
document_topic_df = pd.DataFrame(document_topic, columns=['document_name']+cols)

document_topic_df.to_csv(r'NMF_doctopic_30topics.csv', sep='\t', index=False)

#########################################################################################################################
#t-sne, ELTEC vs Modernity Critique Corpus

num_topics = 30
cols = []
n = 0
while n < num_topics:
    cols.append('topic_' + str(n))
    n+=1

columns=['document_name']+cols
    
document_topic_df = pd.read_csv(r'F:\Kritik_projekt\results\NMF_doctopic_book_30topics_417.csv', sep='\t', names=columns)

meta_df = pd.read_csv(r'F:\Kritik_projekt\3_Korpora_BBAW_normalized\MoL_metadata_20_06_2024_manually_corrected.csv', sep=',')
cols = ['kulturkriCategorical','gesellschaftskriCategorical','zivilisationskriCategorical','dekadenzCategorical','sozialkriCategorical','zeitkriCategorical','modernekriCategorical','heimatkunstCategorical','conservative_revolutionCategorical','worldview_literatureCategorical','fortschrittskriCategorical','filename_normalized_text']#,'adventureCategorical'
meta_df_part = meta_df[cols]

multi_list = []
x = 0
while x < 417:
    row = meta_df_part.loc[[x]]
    true_count = int(row.eq(True).sum(axis=1))
    if true_count > 1:
        multi_list.append(x)
    x+=1

meta_multi_df = meta_df_part.loc[multi_list]
meta_multi_df = meta_multi_df.reset_index()
del meta_multi_df['ELTEC']

multi_file_names = []
orig_file_names = []
row_idx = 0
while row_idx < 90:
    row = meta_multi_df.loc[[row_idx]]
    orig_file_name = row['filename_normalized_text'].item()[:-4]
    orig_file_names.append(orig_file_name)
    for col in cols:
        if row[col].item() == True:
            name = col + '_' + row['filename_normalized_text'].item()[:-4]
            multi_file_names.append(name)
    row_idx +=1

added_rows = []
for orig_name in orig_file_names:
    row = document_topic_df.loc[document_topic_df['document_name'] == 'multi_' + orig_name] 
    for multi_name in multi_file_names:
        if ('_').join(multi_name.split('_')[-3:]) == orig_name:
            new_row = row.replace('multi_' + orig_name, multi_name)
            added_rows.append(new_row)

added_df = pd.concat(added_rows)

document_topic_df = pd.concat([document_topic_df, added_df], ignore_index=True)

group = []
for name in document_topic_df['document_name']:
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

document_topic_df['group'] = group
document_topic_df = document_topic_df.sort_values('group').reset_index(drop = True)
document_topic_df = document_topic_df[document_topic_df.group != 'multiple categories']

t_sne = TSNE(n_components=2, verbose=2, perplexity=5, metric ='cosine', n_iter=300, method='exact', random_state=0)

for_tsne = document_topic_df[document_topic_df.columns[~document_topic_df.columns.isin(['document_name', 'group'])]]
tsne_results = t_sne.fit_transform(for_tsne)

document_topic_df['tsne-2d-one'] = tsne_results[:,0]
document_topic_df['tsne-2d-two'] = tsne_results[:,1]

document_topic_df['style'] = ["ELTeC"] * 81 + ["Modernity Critique Corpus"] * 442

palette = sns.color_palette(cc.glasbey, n_colors=13)

def jitter(values,j):
    return values + np.random.normal(j,0.1,values.shape)

plt.figure(figsize=(12,10))
g = sns.scatterplot(
    x=jitter(document_topic_df["tsne-2d-one"],10), y=jitter(document_topic_df["tsne-2d-two"],10),
    hue="style",
    style="style",
    palette=palette,
    data=document_topic_df,
    legend="full",
    s=250,
    alpha=0.5,
)
g.legend(title='group', loc='upper left')
g.figure.savefig(r'F:\Kritik_projekt\results\Figures\Fig3.jpg',dpi=600, bbox_inches='tight')
plt.show()

#########################################################################################################################
#t-sne, Modernity Critique Corpus only

meta_df = pd.read_csv(r'F:\Kritik_projekt\3_Korpora_BBAW_normalized\MoL_metadata_20_06_2024_manually_corrected.csv', sep=',')
cols = ['ELTEC','kulturkriCategorical','gesellschaftskriCategorical','zivilisationskriCategorical','dekadenzCategorical','sozialkriCategorical','zeitkriCategorical','modernekriCategorical','heimatkunstCategorical','conservative_revolutionCategorical','worldview_literatureCategorical','fortschrittskriCategorical','filename_normalized_text']#,'adventureCategorical'
meta_df_part = meta_df[cols]

multi_list = []
x = 0
while x < 417:
    row = meta_df_part.loc[[x]]
    true_count = int(row.eq(True).sum(axis=1))
    if true_count > 1:
        multi_list.append(x)
    x+=1

meta_multi_df = meta_df_part.loc[multi_list]
meta_multi_df = meta_multi_df.reset_index()

multi_file_names = []
orig_file_names = []
row_idx = 0
while row_idx < 101:
    row = meta_multi_df.loc[[row_idx]]
    orig_file_name = row['filename_normalized_text'].item()[:-4]
    orig_file_names.append(orig_file_name)
    for col in cols:
        if row[col].item() == True:
            name = col + '_' + row['filename_normalized_text'].item()[:-4]
            multi_file_names.append(name)
    row_idx +=1

added_rows = []
for orig_name in orig_file_names:
    row = document_topic_df.loc[document_topic_df['document_name'] == 'multi_' + orig_name] 
    for multi_name in multi_file_names:
        if ('_').join(multi_name.split('_')[-3:]) == orig_name:
            new_row = row.replace('multi_' + orig_name, multi_name)
            added_rows.append(new_row)

added_df = pd.concat(added_rows)

document_topic_df = pd.concat([document_topic_df, added_df], ignore_index=True)

group = []
for name in document_topic_df['document_name']:
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

document_topic_df['group'] = group

document_topic_df = document_topic_df[document_topic_df.group != 'multiple categories']

t_sne = TSNE(n_components=2, verbose=2, perplexity=5, metric ='cosine', n_iter=300, method='exact', random_state=0)#, n_iter_without_progress=100)
for_tsne = document_topic_df[document_topic_df.columns[~document_topic_df.columns.isin(['document_name', 'group'])]]
tsne_results = t_sne.fit_transform(for_tsne)

document_topic_df['tsne-2d-one'] = tsne_results[:,0]
document_topic_df['tsne-2d-two'] = tsne_results[:,1]

document_topic_df = document_topic_df[document_topic_df['group'] != "ELTEC"]

palette = sns.color_palette(cc.glasbey, n_colors=13)

plt.figure(figsize=(18,16))
g = sns.scatterplot(
    x=jitter(document_topic_df["tsne-2d-one"],10), y=jitter(document_topic_df["tsne-2d-two"],10),
    hue="group",
    style="group",
    palette=palette,
    data=document_topic_df,
    legend="full",
    s=250,
    alpha=0.5,
   # markers=markers
)
g.figure.savefig(r'F:\Kritik_projekt\results\Figures\Fig4.jpg',dpi=600, bbox_inches='tight')
plt.show()
g.legend(title='group', loc='best', bbox_to_anchor=(1, 1))

































