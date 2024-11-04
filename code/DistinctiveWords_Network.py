# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:12:39 2024

@author: KeliDu
"""

import pandas as pd
import os
from pyvis.network import Network

df_path = r'F:\Kritik_projekt\most_distinctive_words'
dfs = sorted([os.path.join(df_path, fn) for fn in os.listdir(df_path) if ".csv" in fn])

all_top_words = []
top_n = 10

for df in dfs:
    data = pd.read_csv(df, sep='\t')
    data.rename(columns={'Unnamed: 0':'words'}, inplace=True)
    top_words_idx = []
    idx = 0
    while len(top_words_idx) < top_n:
        word = data.iloc[[idx]]['words'].item()
        if 'noun_' in word: 
            top_words_idx.append(idx)
        if 'adj_' in word:
            top_words_idx.append(idx)
        if 'verb_' in word:
            top_words_idx.append(idx)
        idx += 1
    group_name = os.path.basename(df).split('_')[2]
    for index in top_words_idx:
        all_top_words.append((group_name, data.iloc[[index]]['words'].item(), data.iloc[[index]]['zeta_sd2'].item()))

top_words_df = pd.DataFrame(all_top_words, columns=['text_group', 'top_words', 'zeta_value'])

# replace names
replacement_dict = {
    'zeitkriCategorical': 'critique of the present', 
    'gesellschaftskriCategorical': 'society critique', 
    'sozialkriCategorical': 'social critique', 
    'kulturkriCategorical': 'cultural critique', 
    'modernekriCategorical': 'modernity critique', 
    'zivilisationskriCategorical': 'civilisation critique', 
    'fortschrittskriCategorical': 'progress critique', 
    'WorldviewLiteratureCategorical': 'worldview literature',
    'ConservativeRevolutionCategorical': 'conservative revolution literature',
    'heimatkunstCategorical': 'regional heritage art',
    'dekadenzCategorical': 'decadence literature'
}


# Replace the values in the 'category name' column
top_words_df['text_group'] = top_words_df['text_group'].replace(replacement_dict)

net = Network(
    notebook=True,
    cdn_resources="remote",
    height="1000px",
    width="100%",
    bgcolor="white",
    font_color="black",
    select_menu=True,
)

sources = top_words_df["top_words"]
targets = top_words_df["text_group"]
weights = top_words_df["zeta_value"]

edge_data = zip(sources, targets, weights)

for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]

    net.add_node(src, src, title=src, color='blue', size=15, font={'size': 18})
    net.add_node(dst, dst, title=dst, color='purple', size=28, font={'size': 80})
    net.add_edge(src, dst, value=w, color='pink')

#net.show_buttons(filter_=['physics'])
net.show("network_top10_noun_adj_verb.html")


