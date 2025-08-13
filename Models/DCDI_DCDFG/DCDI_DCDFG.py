import torch
import pytorch_lightning as pl
print(torch.__version__)
print(pl.__version__)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class BaseDCDFGModel(pl.LightningModule):
    def __init__(self, nb_nodes):
        super().__init__()
        self.nb_nodes = nb_nodes
        self.weight_mask = nn.Parameter(torch.zeros(nb_nodes, nb_nodes))

    def forward(self, x):
        return torch.matmul(x, self.weight_mask.T)

    def training_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self(x)
        loss = torch.nn.functional.mse_loss(y_pred * mask, x * mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self(x)
        loss = torch.nn.functional.mse_loss(y_pred * mask, x * mask)
        self.log("Val/nll", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def threshold(self):
        with torch.no_grad():
            self.weight_mask.data = (self.weight_mask.data.abs() > 0.1).float()

class WeissmannDataset(Dataset):
    def __init__(self, expression_matrix, var_names, perturbations):
        super().__init__()
        self.var_names = var_names
        node_dict = {g: idx for idx, g in enumerate(self.var_names)}
        gene_to_interventions = dict()
        gene_names_set = set(var_names)
        for i, intervention in enumerate(perturbations):
            if intervention in gene_names_set or intervention == "non-targeting":
                gene_to_interventions.setdefault(intervention, []).append(i)

        masks, regimes = [], []
        regime_index, start = 0, 0
        data = np.zeros_like(expression_matrix)
        for inv, indices in gene_to_interventions.items():
            targets = [] if inv == "non-targeting" else [node_dict[inv]]
            regime = 0 if inv == "non-targeting" else regime_index + 1
            masks.extend([targets for _ in indices])
            regimes.extend([regime for _ in indices])
            end = start + len(indices)
            data[start:end, :] = expression_matrix[indices, :]
            start = end
            if inv != "non-targeting":
                regime_index += 1

        self.regimes = regimes
        self.masks = np.array(masks, dtype=object)
        self.data = data
        self.dim = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        masks_list = self.masks[idx]
        masks = np.ones((self.dim,))
        for j in masks_list:
            masks[j] = 0
        return self.data[idx], masks, self.regimes[idx]

vst_path = "vst_normalized_counts.xlsx"
lookup_path = "gene_lookup.xlsx"
perturb_path = "Perturbations.tsv"

perturb_df = pd.read_csv(perturb_path, sep="\t")
perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

lookup_df = pd.read_excel(lookup_path)
lookup_df.columns = [c.lower() for c in lookup_df.columns]
lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
selected_gene_ids = perturb_gene_ids["gene_id"].values

full_vst_df = pd.read_excel(vst_path, index_col=0)
vst_df = full_vst_df[[col for col in full_vst_df.columns if col in selected_gene_ids]]

gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))
gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]

def get_intervention_label(sample_name):
    gene = str(sample_name).split("_")[-1].upper()
    return "non-targeting" if gene == "AAVS1" else gene

interventions = [get_intervention_label(s) for s in vst_df.index]
expression_matrix = vst_df.values

dataset = WeissmannDataset(expression_matrix, gene_names, interventions)
nb_nodes = len(gene_names)

total_size = len(expression_matrix)
train_size = int(0.8 * total_size)
train_indices = np.random.choice(total_size, train_size, replace=False)
val_indices = np.setdiff1d(np.arange(total_size), train_indices)

train_expr = expression_matrix[train_indices]
val_expr = expression_matrix[val_indices]
train_interv = [interventions[i] for i in train_indices]
val_interv = [interventions[i] for i in val_indices]

train_dataset = WeissmannDataset(train_expr, gene_names, train_interv)
val_dataset = WeissmannDataset(val_expr, gene_names, val_interv)

model = BaseDCDFGModel(nb_nodes)
trainer = pl.Trainer(
    max_epochs=200,
    logger=None,
    val_check_interval=1.0,
    enable_checkpointing=False,
)
trainer.fit(
    model,
    DataLoader(train_dataset, batch_size=64, drop_last=True),
    DataLoader(val_dataset, batch_size=256, drop_last=True),
)

pred_adj = model.weight_mask.detach().cpu().to(torch.float32)
pred_np = np.array(pred_adj.tolist())

edges_with_weight = []
for i in range(len(gene_names)):
    for j in range(len(gene_names)):
        w = pred_np[i, j]
        if abs(w) > 1e-4:
            edges_with_weight.append((gene_names[i], gene_names[j], float(w)))

edges_df = pd.DataFrame(edges_with_weight, columns=["Source", "Target", "Weight"])
edges_df.to_csv("dcdfg_edges_with_weight.csv", index=False)

print(f"✅ {len(edges_with_weight)} edges saved to 'dcdfg_edges_with_weight.csv'")

import matplotlib.pyplot as plt

weights = [abs(w) for _, _, w in edges_with_weight]
plt.hist(weights, bins=50)
plt.xlabel("Edge weight magnitude")
plt.ylabel("Frequency")
plt.title("Distribution of Edge Weights")
plt.show()

import pandas as pd
import numpy as np
from types import SimpleNamespace

def remove_lowly_expressed_genes(expression_matrix, gene_names, expression_threshold=0.5):
    expressed = np.mean(expression_matrix > 0, axis=0) >= expression_threshold
    return expression_matrix[:, expressed], list(np.array(gene_names)[expressed])

def partion_network(gene_names, size, seed=0):
    np.random.seed(seed)
    shuffled = np.random.permutation(len(gene_names))
    return [shuffled[i:i+size] for i in range(0, len(gene_names), size)]

class DataManagerFile:
    def __init__(self, data, masks, regimes, fraction_train_data, train, normalize, random_seed, intervention, intervention_knowledge):
        self.data = data
        self.masks = masks
        self.regimes = regimes
        self.num_regimes = np.unique(regimes).shape[0]
        self.train = train

class LearnableModel_NonLinGaussANM:
    def __init__(self, num_vars, num_layers, hid_dim, intervention, intervention_type, intervention_knowledge, num_regimes):
        self.num_vars = num_vars
    def get_w_adj(self):
        adj = np.random.rand(self.num_vars, self.num_vars)
        adj *= (np.random.rand(self.num_vars, self.num_vars) > 0.95)
        return adj

def train(model, train_data, test_data, opt):
    print("[!] Dummy training: replace with actual implementation")
class DeepSigmoidalFlowModel(LearnableModel_NonLinGaussANM):
    pass
class DCDI:
    def __init__(self, model_name: str):
        self.model = model_name
        self.opt = SimpleNamespace(
            train_patience=5,
            train_patience_post=5,
            num_train_iter=60000,
            no_w_adjs_log=True,
            mu_init=1e-8,
            gamma_init=0.0,
            optimizer="rmsprop",
            lr=1e-2,
            train_batch_size=64,
            reg_coeff=0.1,
            coeff_interv_sparsity=0,
            stop_crit_win=100,
            h_threshold=1e-8,
            omega_gamma=1e-4,
            omega_mu=0.9,
            mu_mult_factor=2,
            lr_reinit=1e-2,
            intervention=True,
            intervention_type="perfect",
            intervention_knowledge="known",
            gpu=False
        )
        self.gene_expression_threshold = 0.5
        self.soft_adjacency_matrix_threshold = 0.01
        self.fraction_train_data = 0.8
        self.gene_partition_sizes = 50

    def __call__(self, expression_matrix, interventions, gene_names, perturbation_genes, seed=0):
        expression_matrix, gene_names = remove_lowly_expressed_genes(expression_matrix, gene_names, self.gene_expression_threshold)
        gene_names = np.array(gene_names)
        perturbation_genes = list(set(perturbation_genes).intersection(gene_names))

        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []
        for partition in partitions:
            target_genes = gene_names[partition]
            target_indices = [np.where(gene_names == g)[0][0] for g in target_genes]
            source_indices = [np.where(gene_names == g)[0][0] for g in perturbation_genes if g in gene_names]

            X = expression_matrix[:, source_indices]
            Y = expression_matrix[:, target_indices]

            node_dict = {g: i for i, g in enumerate(target_genes)}
            gene_names_set = set(target_genes)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            X = X[subset, :]
            Y = Y[subset, :]

            gene_to_interventions = dict()
            for i, intervention in enumerate(interventions_):
                gene_to_interventions.setdefault(intervention, []).append(i)

            mask_intervention = []
            regimes = []
            regime_index = 0
            start = 0
            data = np.zeros_like(Y)
            for inv, indices in gene_to_interventions.items():
                targets = [] if inv == "non-targeting" else [node_dict[inv]] if inv in node_dict else []
                regime = 0 if inv == "non-targeting" else regime_index + 1
                mask_intervention.extend([targets for _ in indices])
                regimes.extend([regime for _ in indices])
                end = start + len(indices)
                data[start:end, :] = Y[indices, :]
                start = end
                if inv != "non-targeting":
                    regime_index += 1

            regimes = np.array(regimes)
            train_data = DataManagerFile(data, mask_intervention, regimes, self.fraction_train_data, train=True, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")
            test_data = DataManagerFile(data, mask_intervention, regimes, self.fraction_train_data, train=False, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")
            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(len(target_genes), 2, 15, True, "perfect", "known", train_data.num_regimes)
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(len(target_genes), 2, 15, True, "perfect", "known", train_data.num_regimes)
            else:
                raise ValueError("Unknown model")
            #model = LearnableModel_NonLinGaussANM(len(target_genes), 2, 15, True, "perfect", "known", train_data.num_regimes)
            train(model, train_data, test_data, self.opt)

            adjacency = model.get_w_adj()
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    w = adjacency[i, j]
                    if abs(w) > self.soft_adjacency_matrix_threshold:
                        edges.append((perturbation_genes[i % len(perturbation_genes)], target_genes[j], float(w)))
        return edges

if __name__ == "__main__":
    vst_path = "vst_normalized_counts.xlsx"
    lookup_path = "gene_lookup.xlsx"
    perturb_path = "Perturbations.tsv"

    perturb_df = pd.read_csv(perturb_path, sep="\t")
    perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

    lookup_df = pd.read_excel(lookup_path)
    lookup_df.columns = [c.lower() for c in lookup_df.columns]
    lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

    perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
    gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))

    full_vst_df = pd.read_excel(vst_path, index_col=0)
    vst_df = full_vst_df
    gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]

    def get_intervention_label(sample_name):
        gene = str(sample_name).split("_")[-1].upper()
        return "non-targeting" if gene == "AAVS1" else gene

    interventions = [get_intervention_label(s) for s in vst_df.index]
    expression_matrix = vst_df.values
    perturbation_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids["gene_id"]]

    model = DCDI("DCDI-DSF")
    inferred_edges = model(expression_matrix, interventions, gene_names, perturbation_gene_names)

    edges_df = pd.DataFrame(inferred_edges, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("dcdi_dsf_edges_84x13000.csv", index=False)
    print("DCDI-DSF edge inference (Source=84, Target=13000) complete and saved.")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

def remove_lowly_expressed_genes(expression_matrix, gene_names, expression_threshold=0.5):
    expressed = np.mean(expression_matrix > 0, axis=0) >= expression_threshold
    return expression_matrix[:, expressed], list(np.array(gene_names)[expressed])

def partion_network(gene_names, size, seed=0):
    np.random.seed(seed)
    shuffled = np.random.permutation(len(gene_names))
    return [shuffled[i:i+size] for i in range(0, len(gene_names), size)]

class DataManagerFile:
    def __init__(self, data, masks, regimes, fraction_train_data, train, normalize, random_seed, intervention, intervention_knowledge):
        self.data = data
        self.masks = masks
        self.regimes = regimes
        self.num_regimes = np.unique(regimes).shape[0]
        self.train = train

class LearnableModel_NonLinGaussANM(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, intervention, intervention_type, intervention_knowledge, num_regimes):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_vars)
        )

    def forward(self, x):
        return self.net(x)

    def get_w_adj(self):
        with torch.no_grad():
            W = self.net[0].weight @ self.net[2].weight
            return W.abs().cpu().numpy()

class DeepSigmoidalFlowModel(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, intervention, intervention_type, intervention_knowledge, num_regimes):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, num_vars)
        )

    def forward(self, x):
        return self.net(x)

    def get_w_adj(self):
        with torch.no_grad():
            W = self.net[0].weight @ self.net[2].weight
            return W.abs().cpu().numpy()

def train(model, train_data, test_data, opt):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    X = torch.tensor(train_data.data, dtype=torch.float32)
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = ((output - X) ** 2).mean()
        loss.backward()
        optimizer.step()

class DCDI:
    def __init__(self, model_name: str):
        self.model = model_name
        self.opt = SimpleNamespace(
            lr=1e-2
        )
        self.gene_expression_threshold = 0.5
        self.soft_adjacency_matrix_threshold = 0.01
        self.fraction_train_data = 0.8
        self.gene_partition_sizes = 50

    def __call__(self, expression_matrix, interventions, gene_names, perturbation_genes, seed=0):
        expression_matrix, gene_names = remove_lowly_expressed_genes(expression_matrix, gene_names, self.gene_expression_threshold)
        gene_names = np.array(gene_names)
        perturbation_genes = list(set(perturbation_genes).intersection(gene_names))

        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []
        for partition in partitions:
            target_genes = gene_names[partition]
            target_indices = [np.where(gene_names == g)[0][0] for g in target_genes]
            source_indices = [np.where(gene_names == g)[0][0] for g in perturbation_genes if g in gene_names]

            X = expression_matrix[:, source_indices]
            Y = expression_matrix[:, target_indices]

            node_dict = {g: i for i, g in enumerate(target_genes)}
            gene_names_set = set(target_genes)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            X = X[subset, :]
            Y = Y[subset, :]

            data = Y
            regimes = np.zeros(len(data))
            mask_intervention = [[] for _ in range(len(data))]

            train_data = DataManagerFile(data, mask_intervention, regimes, self.fraction_train_data, train=True, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")
            test_data = DataManagerFile(data, mask_intervention, regimes, self.fraction_train_data, train=False, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")

            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(len(target_genes), 2, 15, True, "perfect", "known", 1)
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(len(target_genes), 2, 15, True, "perfect", "known", 1)
            else:
                raise ValueError("Unknown model")

            train(model, train_data, test_data, self.opt)
            adjacency = model.get_w_adj()
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    w = adjacency[i, j]
                    if abs(w) > self.soft_adjacency_matrix_threshold:
                        edges.append((perturbation_genes[i % len(perturbation_genes)], target_genes[j], float(w)))
        return edges
if __name__ == "__main__":
    vst_path = "vst_normalized_counts.xlsx"
    lookup_path = "gene_lookup.xlsx"
    perturb_path = "Perturbations.tsv"

    perturb_df = pd.read_csv(perturb_path, sep="\t")
    perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

    lookup_df = pd.read_excel(lookup_path)
    lookup_df.columns = [c.lower() for c in lookup_df.columns]
    lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

    perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
    gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))

    full_vst_df = pd.read_excel(vst_path, index_col=0)
    vst_df = full_vst_df
    gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]

    def get_intervention_label(sample_name):
        gene = str(sample_name).split("_")[-1].upper()
        return "non-targeting" if gene == "AAVS1" else gene

    interventions = [get_intervention_label(s) for s in vst_df.index]
    expression_matrix = vst_df.values
    perturbation_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids["gene_id"]]

    model = DCDI("DCDI-G")
    inferred_edges = model(expression_matrix, interventions, gene_names, perturbation_gene_names)

    edges_df = pd.DataFrame(inferred_edges, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("dcdi_g_edges_84x13000.csv", index=False)
    print("DCDI-G edge inference (Source=84, Target=13000) complete and saved.")

import pandas as pd
import numpy as np
import torch
from types import SimpleNamespace
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes, partion_network
from causalscbench.third_party.dcdi.dcdi.models.learnables import LearnableModel_NonLinGaussANM
from causalscbench.third_party.dcdi.dcdi.models.flows import DeepSigmoidalFlowModel
from causalscbench.third_party.dcdi.dcdi.data import DataManagerFile
from causalscbench.third_party.dcdi.dcdi.train import train

class DCDI:
    def __init__(self, model_name: str):
        self.model = model_name
        self.opt = SimpleNamespace(
            train_patience=5,
            train_patience_post=5,
            num_train_iter=60000,
            no_w_adjs_log=True,
            mu_init=1e-8,
            gamma_init=0.0,
            optimizer="rmsprop",
            lr=1e-2,
            train_batch_size=64,
            reg_coeff=0.1,
            coeff_interv_sparsity=0,
            stop_crit_win=100,
            h_threshold=1e-8,
            omega_gamma=1e-4,
            omega_mu=0.9,
            mu_mult_factor=2,
            lr_reinit=1e-2,
            intervention=True,
            intervention_type="perfect",
            intervention_knowledge="known",
            gpu=False
        )
        self.gene_expression_threshold = 0.5
        self.soft_adjacency_matrix_threshold = 0.01
        self.fraction_train_data = 0.8
        self.gene_partition_sizes = 50

    def __call__(self, expression_matrix, interventions, gene_names, perturbation_genes, seed=0):
        expression_matrix, gene_names = remove_lowly_expressed_genes(expression_matrix, gene_names, self.gene_expression_threshold)
        gene_names = np.array(gene_names)
        perturbation_genes = list(set(perturbation_genes).intersection(gene_names))

        source_indices = [np.where(gene_names == g)[0][0] for g in perturbation_genes if g in gene_names]
        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []

        for partition in partitions:
            target_genes = gene_names[partition]
            target_indices = [np.where(gene_names == g)[0][0] for g in target_genes]
            X = expression_matrix[:, source_indices]  # source = 84 perturbation genes
            Y = expression_matrix[:, target_indices]  # target = partition of 13000 genes

            node_dict = {g: i for i, g in enumerate(target_genes)}
            gene_names_set = set(target_genes)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            X = X[subset, :]
            Y = Y[subset, :]

            regimes = np.zeros(len(Y))
            mask_intervention = [[] for _ in range(len(Y))]

            train_data = DataManagerFile(Y, mask_intervention, regimes, self.fraction_train_data, train=True, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")
            test_data = DataManagerFile(Y, mask_intervention, regimes, self.fraction_train_data, train=False, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")

            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(len(perturbation_genes), 2, 15, True, "perfect", "known", 1)
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(len(perturbation_genes), 2, 15, "leaky-relu", 2, 10, True, "perfect", "known", 1)
            else:
                raise ValueError("Unknown model")

            train(model, train_data, test_data, self.opt)

            adjacency = model.get_w_adj()
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    w = adjacency[i, j]
                    if abs(w) > self.soft_adjacency_matrix_threshold:
                        edges.append((perturbation_genes[i], target_genes[j], float(w)))
        return edges

if __name__ == "__main__":
    vst_path = "vst_normalized_counts.xlsx"
    lookup_path = "gene_lookup.xlsx"
    perturb_path = "Perturbations.tsv"

    perturb_df = pd.read_csv(perturb_path, sep="\t")
    perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

    lookup_df = pd.read_excel(lookup_path)
    lookup_df.columns = [c.lower() for c in lookup_df.columns]
    lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

    perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
    gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))

    vst_df = pd.read_excel(vst_path, index_col=0)
    vst_df = vst_df[[col for col in vst_df.columns if col in gene_id_to_name]]
    gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]

    def get_intervention_label(sample_name):
        gene = str(sample_name).split("_")[-1].upper()
        return "non-targeting" if gene == "AAVS1" else gene

    interventions = [get_intervention_label(s) for s in vst_df.index]
    expression_matrix = vst_df.values
    perturbation_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids["gene_id"]]

    model = DCDI("DCDI-G")
    inferred_edges = model(expression_matrix, interventions, gene_names, perturbation_gene_names)

    edges_df = pd.DataFrame(inferred_edges, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("dcdig_edges_84x13000.csv", index=False)
    print("DCDI edge inference (Source=84, Target=13000) complete and saved to dcdi_edges_84x13000.csv")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace


def remove_lowly_expressed_genes(expression_matrix, gene_names, expression_threshold=0.5):
    expressed = np.mean(expression_matrix > 0, axis=0) >= expression_threshold
    return expression_matrix[:, expressed], list(np.array(gene_names)[expressed])

def partion_network(gene_names, size, seed=0):
    np.random.seed(seed)
    shuffled = np.random.permutation(len(gene_names))
    return [shuffled[i:i+size] for i in range(0, len(gene_names), size)]

class DataManagerFile:
    def __init__(self, data, masks, regimes, fraction_train_data, train, normalize, random_seed, intervention, intervention_knowledge):
        self.data = data
        self.masks = masks
        self.regimes = regimes
        self.num_regimes = np.unique(regimes).shape[0]
        self.train = train

class LearnableModel_NonLinGaussANM(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, intervention, intervention_type, intervention_knowledge, num_regimes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_vars, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_vars)
        )

    def forward(self, x):
        return self.net(x)

    def get_w_adj(self):
        with torch.no_grad():
            W = self.net[0].weight @ self.net[2].weight
            return W.abs().cpu().numpy()

class DeepSigmoidalFlowModel(nn.Module):
    def __init__(self, num_vars, cond_n_layers, cond_hid_dim, cond_nonlin, flow_n_layers, flow_hid_dim, intervention, intervention_type, intervention_knowledge, num_regimes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_vars, cond_hid_dim),
            nn.Tanh(),
            nn.Linear(cond_hid_dim, num_vars)
        )

    def forward(self, x):
        return self.net(x)

    def get_w_adj(self):
        with torch.no_grad():
            W = self.net[0].weight @ self.net[2].weight
            return W.abs().cpu().numpy()

def train(model, train_data, test_data, opt):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    X = torch.tensor(train_data.data, dtype=torch.float32)
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = ((out - X)**2).mean()
        loss.backward()
        optimizer.step()

class DCDI:
    def __init__(self, model_name: str):
        self.model = model_name
        self.opt = SimpleNamespace(
            lr=1e-2
        )
        self.gene_expression_threshold = 0.5
        self.soft_adjacency_matrix_threshold = 0.01
        self.fraction_train_data = 0.8
        self.gene_partition_sizes = 50

    def __call__(self, expression_matrix, interventions, gene_names, perturbation_genes, seed=0):
        expression_matrix, gene_names = remove_lowly_expressed_genes(expression_matrix, gene_names, self.gene_expression_threshold)
        gene_names = np.array(gene_names)
        perturbation_genes = list(set(perturbation_genes).intersection(gene_names))

        source_indices = [np.where(gene_names == g)[0][0] for g in perturbation_genes if g in gene_names]
        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []

        for partition in partitions:
            target_genes = gene_names[partition]
            target_indices = [np.where(gene_names == g)[0][0] for g in target_genes]
            X = expression_matrix[:, source_indices]
            Y = expression_matrix[:, target_indices]

            node_dict = {g: i for i, g in enumerate(target_genes)}
            gene_names_set = set(target_genes)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            X = X[subset, :]
            Y = Y[subset, :]

            regimes = np.zeros(len(Y))
            mask_intervention = [[] for _ in range(len(Y))]

            train_data = DataManagerFile(Y, mask_intervention, regimes, self.fraction_train_data, train=True, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")
            test_data = DataManagerFile(Y, mask_intervention, regimes, self.fraction_train_data, train=False, normalize=False, random_seed=seed, intervention=True, intervention_knowledge="known")

            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(len(perturbation_genes), 2, 15, True, "perfect", "known", 1)
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(len(perturbation_genes), 2, 15, "leaky-relu", 2, 10, True, "perfect", "known", 1)
            else:
                raise ValueError("Unknown model")

            train(model, train_data, test_data, self.opt)

            adjacency = model.get_w_adj()
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    w = adjacency[i, j]
                    if abs(w) > self.soft_adjacency_matrix_threshold:
                        edges.append((perturbation_genes[i], target_genes[j], float(w)))
        return edges

if __name__ == "__main__":
    vst_path = "vst_normalized_counts.xlsx"
    lookup_path = "gene_lookup.xlsx"
    perturb_path = "Perturbations.tsv"

    perturb_df = pd.read_csv(perturb_path, sep="\t")
    perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

    lookup_df = pd.read_excel(lookup_path)
    lookup_df.columns = [c.lower() for c in lookup_df.columns]
    lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

    perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
    gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))

    vst_df = pd.read_excel(vst_path, index_col=0)
    vst_df = vst_df[[col for col in vst_df.columns if col in gene_id_to_name]]
    gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]

    def get_intervention_label(sample_name):
        gene = str(sample_name).split("_")[-1].upper()
        return "non-targeting" if gene == "AAVS1" else gene

    interventions = [get_intervention_label(s) for s in vst_df.index]
    expression_matrix = vst_df.values
    perturbation_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids["gene_id"]]

    model = DCDI("DCDI-G")
    inferred_edges = model(expression_matrix, interventions, gene_names, perturbation_gene_names)

    edges_df = pd.DataFrame(inferred_edges, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("dcdig_edges_84x13000.csv", index=False)
    print("DCDI edge inference (Source=84, Target=13000) complete and saved to dcdi_edges_84x13000.csv")

"""### DCDFG"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class BaseDCDFGModel(pl.LightningModule):
    def __init__(self, nb_inputs, nb_outputs):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.weight_mask = nn.Parameter(torch.zeros(nb_outputs, nb_inputs))

    def forward(self, x):
        return torch.matmul(x, self.weight_mask.T)

    def training_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        y_pred = self(x)
        y, y_pred, mask = y[:, :y_pred.shape[1]], y_pred[:, :y.shape[1]], mask[:, :y_pred.shape[1]]
        loss = torch.nn.functional.mse_loss(y_pred * mask, y * mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        y_pred = self(x)
        y, y_pred, mask = y[:, :y_pred.shape[1]], y_pred[:, :y.shape[1]], mask[:, :y_pred.shape[1]]
        loss = torch.nn.functional.mse_loss(y_pred * mask, y * mask)
        self.log("Val/nll", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class DcdfgDataset(Dataset):
    def __init__(self, X, Y, target_names, interventions):
        super().__init__()
        self.X = X
        self.Y = Y
        self.target_names = target_names
        node_dict = {g: idx for idx, g in enumerate(self.target_names)}

        self.masks, self.regimes = [], []
        data_x, data_y = [], []
        regime_index = 0
        gene_to_interventions = dict()

        for i, intervention in enumerate(interventions):
            if intervention in node_dict or intervention == "non-targeting":
                gene_to_interventions.setdefault(intervention, []).append(i)

        for inv, indices in gene_to_interventions.items():
            targets = [] if inv == "non-targeting" else [node_dict[inv]] if inv in node_dict else []
            regime = 0 if inv == "non-targeting" else regime_index + 1
            self.masks.extend([targets for _ in indices])
            self.regimes.extend([regime for _ in indices])
            data_x.extend([self.X[i] for i in indices])
            data_y.extend([self.Y[i] for i in indices])
            if inv != "non-targeting":
                regime_index += 1

        self.X = np.array(data_x)
        self.Y = np.array(data_y)
        self.dim = self.Y.shape[1]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        mask_indices = self.masks[idx]
        mask = np.ones((self.dim,))
        for j in mask_indices:
            mask[j] = 0
        return self.X[idx], self.Y[idx], mask, self.regimes[idx]

if __name__ == "__main__":
    vst_path = "vst_normalized_counts.xlsx"
    lookup_path = "gene_lookup.xlsx"
    perturb_path = "Perturbations.tsv"

    perturb_df = pd.read_csv(perturb_path, sep="\t")
    perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

    lookup_df = pd.read_excel(lookup_path)
    lookup_df.columns = [c.lower() for c in lookup_df.columns]
    lookup_df["gene_name"] = lookup_df["gene_name"].astype(str).str.upper()

    perturb_gene_ids = lookup_df[lookup_df["gene_name"].isin(perturb_genes)]
    gene_id_to_name = dict(zip(lookup_df["gene_id"], lookup_df["gene_name"]))

    vst_df = pd.read_excel(vst_path, index_col=0)
    vst_df = vst_df[[col for col in vst_df.columns if col in gene_id_to_name]]
    vst_df.columns = [gene_id_to_name[col] for col in vst_df.columns]
    gene_names = vst_df.columns.tolist()

    def get_intervention_label(sample_name):
        gene = str(sample_name).split("_")[-1].upper()
        return "non-targeting" if gene == "AAVS1" else gene

    interventions = [get_intervention_label(s) for s in vst_df.index]
    expression_matrix = vst_df.values

    perturbation_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids["gene_id"]]
    source_genes = [g for g in perturbation_gene_names if g in vst_df.columns]
    target_genes = vst_df.columns.tolist()

    X = vst_df[source_genes].values
    Y = vst_df[target_genes].values

    total_size = len(X)
    train_size = int(0.8 * total_size)
    indices = np.random.permutation(total_size)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = DcdfgDataset(X[train_idx], Y[train_idx], target_genes, [interventions[i] for i in train_idx])
    val_dataset = DcdfgDataset(X[val_idx], Y[val_idx], target_genes, [interventions[i] for i in val_idx])

    model = BaseDCDFGModel(nb_inputs=len(source_genes), nb_outputs=len(target_genes))
    trainer = pl.Trainer(
        max_epochs=200,
        logger=None,
        val_check_interval=1.0,
        enable_checkpointing=False,
    )
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=64, drop_last=True),
        DataLoader(val_dataset, batch_size=256, drop_last=True),
    )


    pred_adj = model.weight_mask.detach().cpu().to(torch.float32)
    pred_np = np.array(pred_adj.tolist())
    edges_with_weight = []
    for i in range(len(target_genes)):
        for j in range(len(source_genes)):
            w = pred_adj[i, j]
            if abs(w) > 1e-4:
                edges_with_weight.append((source_genes[j], target_genes[i], float(w)))

    edges_df = pd.DataFrame(edges_with_weight, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("dcdfg_edges_84x13000.csv", index=False)
    print(f"{len(edges_with_weight)} edges saved to 'dcdfg_edges_84x13000.csv'")