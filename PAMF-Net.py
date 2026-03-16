'Written by Cagri Ozdemir, PhD on 12/08/2025'

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
import gseapy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from library.model import test_v1

############################
# CONFIG
############################

FOLDS = 5
HID_SIZE1 = 128
HID_SIZE2 = 64
OUT_DIM = 1
LR = 0.003
EPOCHS = 2000
MIN_EPOCHS = 50
PATIENCE = 30

BASE_PATH = ''
DATA_DIR = os.path.join(BASE_PATH,'datasets')
library_name0 = "MSigDB_Hallmark_2020"
library_name1 = "KEGG_2021_Human"
DATASET = 'BRCA_data' # 'METABRIC_data' TODO select the data
device = torch.device("cpu")


############################
# DATA LOADING
############################

def load_data(dataset=DATASET, min_genes=5, max_genes=550):

    snv_data = pd.read_csv(os.path.join(DATA_DIR, dataset, 'snv_data.csv'))
    clinical_data = torch.load(os.path.join(DATA_DIR, dataset,'clinical_onehot_tensor.pt'))
    bc_recurrence_pathways = torch.load(os.path.join(DATA_DIR, dataset,'bc_recurrence_pathways.pt'))
    response = pd.read_csv(os.path.join(DATA_DIR,dataset,'response.csv'))
    labels = torch.tensor(response.values, dtype=torch.long)[:,0]

    df_rs0 = snv_data.reset_index().iloc[:,2:]
    snv_data_npy = df_rs0.to_numpy()

    gene_names = snv_data.columns.astype(str)[1:]
    gene_to_idx = {g:i for i,g in enumerate(gene_names)}

    hallmark0 = gseapy.get_library(name=library_name0, organism="Human")
    hallmark1 = gseapy.get_library(name=library_name1, organism="Human")
    hallmark = {**hallmark0, **hallmark1}

    selected_hallmark = {
        pname: hallmark[pname]
        for pname in bc_recurrence_pathways
        if pname in hallmark
    }

    pathways = {}

    for pname, genes in selected_hallmark.items():

        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]

        if min_genes <= len(idx) <= max_genes:
            pathways[pname] = np.array(idx, dtype=int)

    data_dic = {}

    for i, pname in enumerate(pathways):
        data_dic[i] = torch.tensor(
            snv_data_npy[:, pathways[pname]],
            dtype=torch.float32
        )

    return data_dic, labels.float(), clinical_data, pathways


############################
# TRAIN / VALIDATION
############################

def train_epoch(model, data_dic, clinical_data, labels, mask, optimizer, criterion):

    model.train()

    optimizer.zero_grad()

    weights, out = model(data_dic, clinical_data)

    reg_lambda = 1e-3
    scores_flat = weights.squeeze(-1)

    att_reg = reg_lambda * torch.sum(scores_flat**2) / scores_flat.numel()

    loss = criterion(out[mask], labels[mask]) + att_reg

    loss.backward()
    optimizer.step()

    return loss.item()


def validate_epoch(model, data_dic, clinical_data, labels, mask, criterion):

    model.eval()

    with torch.no_grad():

        weights, out = model(data_dic, clinical_data)

        reg_lambda = 1e-3
        scores_flat = weights.squeeze(-1)

        att_reg = reg_lambda * torch.sum(scores_flat**2) / scores_flat.numel()

        loss = criterion(out[mask], labels[mask]) + att_reg

    return loss.item()


############################
# TRAIN ONE FOLD
############################

def train_fold(fold, train_idx, test_idx, data_dic, labels, clinical_data, modalities):

    print(f"Fold {fold}")

    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.25,
        stratify=labels[train_idx],
        shuffle=True,
        random_state=1029
    )

    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    val_mask = torch.zeros_like(labels, dtype=torch.bool)
    test_mask = torch.zeros_like(labels, dtype=torch.bool)

    train_mask[train_sub_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True


    model = test_v1(
        modalities,
        hid_dim1=HID_SIZE1,
        hid_dim11=HID_SIZE1 + clinical_data.shape[1],
        hid_dim2=HID_SIZE2,
        out_dim=OUT_DIM
    ).to(device)


    optimizer = Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    min_valid_loss = np.inf
    patience_count = 0


    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model, data_dic, clinical_data,
            labels, train_mask,
            optimizer, criterion
        )

        val_loss = validate_epoch(
            model, data_dic, clinical_data,
            labels, val_mask,
            criterion
        )

        if val_loss < min_valid_loss:

            min_valid_loss = val_loss
            patience_count = 0

        else:

            patience_count += 1

        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break


    return model, test_idx


############################
# CROSS VALIDATION
############################

def cross_validate(data_dic, labels, clinical_data):

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1029)

    modalities = {k:v.shape[1] for k,v in data_dic.items()}

    metrics = {'acc':[], 'f1':[], 'auc':[], 'aupr':[]}

    weights_list = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_dic[0], labels)):

        model, test_idx = train_fold(
            fold, train_idx, test_idx,
            data_dic, labels, clinical_data,
            modalities
        )

        model.eval()

        with torch.no_grad():

            weights_all, out_all = model(data_dic, clinical_data)

        weights_list[fold] = torch.mean(weights_all,0).squeeze(1)

        preds_test = (out_all[test_idx] > 0.5).int().cpu().numpy()
        gt_labels = labels[test_idx].cpu().numpy()
        probs_test = out_all[test_idx].cpu().numpy()

        metrics['acc'].append(accuracy_score(gt_labels, preds_test))
        metrics['f1'].append(f1_score(gt_labels, preds_test))
        metrics['auc'].append(roc_auc_score(gt_labels, probs_test))
        metrics['aupr'].append(average_precision_score(gt_labels, probs_test))

    return metrics, weights_list


############################
# PATHWAY IMPORTANCE PLOT
############################

def plot_pathway_importance(weights_list, pathways):

    all_weights = torch.stack(list(weights_list.values()), dim=0)

    average_weights = torch.mean(all_weights, dim=0).squeeze()

    pathway_names = list(pathways.keys())

    avg_weights_np = average_weights.detach().cpu().numpy()

    percentages = (avg_weights_np / np.sum(avg_weights_np)) * 100

    sorted_idx = np.argsort(percentages)[::-1]

    sorted_pathways = [pathway_names[i] for i in sorted_idx]
    sorted_percentages = [percentages[i] for i in sorted_idx]

    top_n = 15

    top_pathways = sorted_pathways[:top_n]
    top_percentages = sorted_percentages[:top_n]

    top_pathways.append("Others")
    top_percentages.append(sum(sorted_percentages[top_n:]))

    plt.figure(figsize=(12,8),dpi=130)

    bars = plt.barh(top_pathways, top_percentages, color="skyblue")

    plt.xlabel("Attention (%)", fontsize=15)
    plt.title("METABRIC: Pathway Importance", fontsize=16)

    plt.gca().invert_yaxis()

    for bar, pct in zip(bars, top_percentages):

        plt.text(
            bar.get_width()+0.5,
            bar.get_y()+bar.get_height()/2,
            f"{pct:.2f}%",
            va="center",
            fontsize=13
        )

    plt.tight_layout()
    plt.show()


############################
# MAIN
############################

def main():

    data_dic, labels, clinical_data, pathways = load_data()

    torch.manual_seed(44)

    metrics, weights_list = cross_validate(
        data_dic,
        labels,
        clinical_data
    )

    print(f"ACC : {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
    print(f"F1  : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    print(f"AUC : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
    print(f"AUPR: {np.mean(metrics['aupr']):.4f} ± {np.std(metrics['aupr']):.4f}")

    plot_pathway_importance(weights_list, pathways)


if __name__ == "__main__":

    main()


