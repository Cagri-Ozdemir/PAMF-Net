'Written by Cagri Ozdemir,PhD on 12/08/2025'
import os
import pandas as pd
import numpy as np
import matplotlib
import torch
matplotlib.use('TkAgg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import gseapy

BASE_PATH = ''
DATASET = 'BRCA_data' # 'METABRIC_data' TODO select the data
DATA_DIR = os.path.join(BASE_PATH,'datasets')
library_name0= "MSigDB_Hallmark_2020"
library_name1= "KEGG_2021_Human"

def load_data(dataset='data', min=5, max= 550):
    snv_data = pd.read_csv(os.path.join(DATA_DIR, dataset, 'snv_data.csv'))
    clinical_data = torch.load(os.path.join(DATA_DIR, dataset,'clinical_onehot_tensor.pt'))
    bc_recurrence_pathways = torch.load(os.path.join(DATA_DIR, dataset,'bc_recurrence_pathways.pt'))
    response = pd.read_csv(os.path.join(DATA_DIR,dataset,'response.csv'))
    labels = torch.tensor(response.values, dtype=torch.long)[:,0]
    df_rs0 = snv_data.reset_index().iloc[:, 2:]
    snv_data_npy = df_rs0.to_numpy()
    gene_names = snv_data.columns.astype(str)[1:]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    hallmark0 = gseapy.get_library(name=library_name0, organism="Human")
    hallmark1 = gseapy.get_library(name=library_name1, organism="Human")
    hallmark = {**hallmark0, **hallmark1}
    selected_hallmark = {
        pname: hallmark[pname]
        for pname in bc_recurrence_pathways
        if pname in hallmark
    }
    g_n ={}
    pathways = {}
    for pname, genes in selected_hallmark.items():
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if min <= len(idx) <= max:
            pathways[pname] = np.array(idx, dtype=int)
            g_n[pname] = gene_names[idx]
    data_dic = {}

    for i, pname in enumerate(pathways):
        data_dic[i] = torch.tensor(snv_data_npy[:, pathways[pname]], dtype=torch.float32)
    return data_dic, labels, clinical_data, pathways,g_n

data_dic, labels, clinical_data, pathways,g_n = load_data(dataset=DATASET)
labels = labels.float()


labels = np.array(labels)  # Ensure numpy array
name = list(g_n.keys())
path_num = 16  # ( path_num is 15 for central carbon, 6 for regulation of actin, 14 for miRNA, and 8 for JAK-SAT)
if DATASET=='BRCA_data':
    path_num = path_num + 1
data = np.array(data_dic[path_num])
names_genes = g_n[name[path_num]]  # List of gene names

n_samples, n_genes = data.shape
log2_or = []

# Compute log2(OR) for each gene
for g in range(n_genes):
    X = (data[:, g] > 0).astype(int)

    table = np.array([
        [np.sum((X==1) & (labels==1)), np.sum((X==1) & (labels==0))],
        [np.sum((X==0) & (labels==1)), np.sum((X==0) & (labels==0))]
    ])

    # Add pseudo-count to avoid infinite OR
    table_safe = table + 0.5
    oddsratio = (table_safe[0,0]*table_safe[1,1]) / (table_safe[0,1]*table_safe[1,0])
    log2_or.append(np.log2(oddsratio))

log2_or = np.array(log2_or)

# Masks
pos_mask = log2_or > 0
neg_mask = log2_or < 0

# Create color maps
pos_cmap = plt.get_cmap('Reds')      # Light → dark red
neg_cmap = plt.get_cmap('Blues_r')   # Light → dark blue

# Normalize separately
pos_norm = mcolors.Normalize(vmin=0, vmax=log2_or[pos_mask].max() if pos_mask.any() else 1)
neg_norm = mcolors.Normalize(vmin=log2_or[neg_mask].min() if neg_mask.any() else -1, vmax=0)

# Assign gradient colors
colors = []
for i in range(n_genes):
    if pos_mask[i]:
        colors.append(pos_cmap(pos_norm(log2_or[i])))
    elif neg_mask[i]:
        colors.append(neg_cmap(neg_norm(log2_or[i])))
    else:
        colors.append('grey')  # OR = 0

# Plot lollipop
plt.figure(figsize=(12,8),dpi=130)
for i in range(n_genes):
    plt.plot([i, i], [0, log2_or[i]], color=colors[i], linewidth=3)  # stick
    plt.scatter(i, log2_or[i], color=colors[i], s=50)                 # candy
    # Gene name colored same as dot
    plt.text(i, log2_or[i] + 0.2 * np.sign(log2_or[i]), names_genes[i],
             fontsize=15, ha='center', va='bottom' if log2_or[i]>0 else 'top',
             rotation=45, color=colors[i])

plt.axhline(0, color='grey', linestyle='--')
plt.ylabel('log2(Odds Ratio)',fontsize=15)
plt.ylim(-4,6)
plt.title(f"{DATASET}: SNVs enrichment in {name[path_num]} pathway", fontsize=16)
plt.xticks([])
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()
