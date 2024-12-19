# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 05:00:21 2024

@author: KeliDu
"""

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#get 6 class classification results, create boxplot for the comparison between ELTeC and Modernity Critique Corpus
import os
import pandas as pd
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
import colorcet as cc

folder_class6 = r'F:\Kritik_projekt\corpus_classification'
files_class6 = sorted([os.path.join(folder_class6, fn) for fn in os.listdir(folder_class6)])

all_count_dfs = []
file_count = 0
while file_count < len(files_class6):
    df = pd.read_csv(files_class6[file_count], sep='\t')
    count = Counter(df['dimension'])
    count_sum = sum(count.values())
    count_df = pd.DataFrame.from_dict(count, orient='index').reset_index()
    count_df = count_df.rename(columns={'index':'dimension', 0:'count'})
    count_df['group'] = [os.path.basename(files_class6[file_count]).split('_')[0]] * len(count_df)
    count_df['percentage'] = count_df['count'].div(count_sum)
    all_count_dfs.append(count_df)
    file_count+=1

all_counts = pd.concat(all_count_dfs)

all_counts['group'] = all_counts['group'].map({'zeitkriCategorical': 'critique of the present', 
                                               'gesellschaftskriCategorical': 'society critique',
                                               'sozialkriCategorical': 'social critique',
                                               'kulturkriCategorical': 'cultural critique',
                                               'modernekriCategorical':'modernity critique',
                                               'zivilisationskriCategorical':'civilisation critique',
                                               'fortschrittskriCategorical':'progress critique',
                                               'dekadenzCategorical':'decadence literature',
                                               'heimatkunstCategorical':'regional heritage art',
                                               'worldview':'worldview literature',
                                               'conservative':'conservative revolution literature',
                                               'ELTEC':'ELTeC',
                                               'multi':'multiple categories'})

all_counts['dimension'] = all_counts['dimension'].map({'Gesund-Krank': 'Healthy-ill', 
                                               'Harmonisch-Disharmonisch': 'Harmonious-Disharmonious',
                                               'Natürlich-Kulturell': 'Natural-Cultural',
                                               'Tief-Oberflächlich': 'Profound-Superficial',
                                               'Traditionell-Modern':'Traditional-Modern'})

all_counts = all_counts.fillna('None')
all_counts['class'] = ["ELTeC"] * 486 + ["Modernity Critique Corpus"] * 1988
df_visual = all_counts[all_counts['dimension'] != 'None']

palette = sns.color_palette(cc.glasbey, n_colors=13)
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (16,8))
g = sns.boxplot(x='dimension', y='percentage', hue='class', palette=palette, data=df_visual)#, showfliers=False)
g.legend(title='group', loc='best', bbox_to_anchor=(1, 1))
g.figure.savefig(r'F:\Kritik_projekt\results\Figures\Fig10.jpg',dpi=600)
plt.show()

df_visual_1 = df_visual.loc[~df_visual['group'].isin(['social critique', 'society critique'])]
palette = sns.color_palette(cc.glasbey, n_colors=13)
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (16,8))
g = sns.boxplot(x='dimension', y='percentage', hue='class', palette=palette, data=df_visual_1)#, showfliers=False)
g.legend(title='group', loc='best', bbox_to_anchor=(1, 1))
g.figure.savefig(r'F:\Kritik_projekt\results\Figures\Fig10_.jpg',dpi=600)
plt.show()

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#get 6 class classification results, create boxplot for the comparison between ELTeC and single modernity groups
import os
import pandas as pd
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
import colorcet as cc

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

folder_class6 = r'F:\Kritik_projekt\corpus_classification'
files_class6 = sorted([os.path.join(folder_class6, fn) for fn in os.listdir(folder_class6)])

all_count_dfs = []
file_count = 0
while file_count < len(files_class6):
    df = pd.read_csv(files_class6[file_count], sep='\t')
    file_name = os.path.basename(files_class6[file_count])
    count = Counter(df['dimension'])
    count_sum = sum(count.values())
    count_df = pd.DataFrame.from_dict(count, orient='index').reset_index()
    count_df = count_df.rename(columns={'index':'dimension', 0:'count'})
    if file_name[:-4].split('_')[0] == 'multi':
        for i in multi_file_names:
            if '_'.join(file_name[:-4].split('_')[1:]) in i:
                print(i)
                count_df_i = count_df.copy()
                count_df_i['group'] = [i.split('_')[0]] * len(count_df)
                count_df_i['document_name'] = file_name[:-4]
                count_df_i['average percentage'] = count_df_i['count'].div(count_sum)
                all_count_dfs.append(count_df_i)
    if file_name[:-4].split('_')[0] != 'multi':            
        count_df['group'] = [file_name.split('_')[0]] * len(count_df)
        count_df['document_name'] = file_name[:-4]
        count_df['average percentage'] = count_df['count'].div(count_sum)
        all_count_dfs.append(count_df)
    file_count+=1

all_counts = pd.concat(all_count_dfs)

all_counts['group'] = all_counts['group'].map({'zeitkriCategorical': 'critique of the present', 
                                               'gesellschaftskriCategorical': 'society critique',
                                               'sozialkriCategorical': 'social critique',
                                               'kulturkriCategorical': 'cultural critique',
                                               'modernekriCategorical':'modernity critique',
                                               'zivilisationskriCategorical':'civilisation critique',
                                               'fortschrittskriCategorical':'progress critique',
                                               'dekadenzCategorical':'decadence literature',
                                               'heimatkunstCategorical':'regional heritage art',
                                               'worldview':'worldview literature',
                                               'conservative':'conservative revolution literature',
                                               'ELTEC':'ELTeC',
                                               'multi':'multiple categories'})

all_counts['dimension'] = all_counts['dimension'].map({'Gesund-Krank': 'Healthy-ill', 
                                               'Harmonisch-Disharmonisch': 'Harmonious-Disharmonious',
                                               'Natürlich-Kulturell': 'Natural-Cultural',
                                               'Tief-Oberflächlich': 'Profound-Superficial',
                                               'Traditionell-Modern':'Traditional-Modern'})

all_counts = all_counts.fillna('None')

all_counts_avg = []
for group in set(all_counts['group']):
    df_one_group = all_counts[all_counts['group'] == group]
    df_one_group = df_one_group[['dimension', 'average percentage']]
    df_one_group1 = df_one_group.groupby(['dimension']).mean()
    df_one_group1['group'] = [group] * len(df_one_group1)
    all_counts_avg.append(df_one_group1)

all_counts_avg_df = pd.concat(all_counts_avg)
all_counts_avg_df = all_counts_avg_df.reset_index()
all_counts_avg_df_visual = all_counts_avg_df[all_counts_avg_df['dimension'] != 'None']
all_counts_avg_df_visual = all_counts_avg_df_visual.sort_values(['group', 'dimension'])

pd.DataFrame.iteritems = pd.DataFrame.items
import seaborn.objects as so


(
 so.Plot(all_counts_avg_df_visual, x='average percentage', y="group", color="dimension")
.add(so.Bar(), so.Stack())
.layout(size=(12, 7))
.limit(x=(0,0.15))
.scale(color='viridis')#icefire
.save(r'F:\Kritik_projekt\results\Figures\Fig111.jpg',dpi=600, bbox_inches='tight')
)
