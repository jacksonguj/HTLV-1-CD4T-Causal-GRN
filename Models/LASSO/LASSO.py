# import module
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# file directory
vst_path = "vst_normalized_counts.xlsx"
lookup_path = "gene_lookup.xlsx"
perturb_path = "Perturbations.tsv"

lookup_df = pd.read_excel(lookup_path)
lookup_df.columns = [c.lower() for c in lookup_df.columns]
lookup_df['gene_name'] = lookup_df['gene_name'].astype(str).str.upper()

perturb_df = pd.read_csv(perturb_path, sep="\t", header=None)
perturb_genes = perturb_df.iloc[:, 0].astype(str).str.upper().unique()

perturb_gene_ids = lookup_df[lookup_df['gene_name'].isin(perturb_genes)]
perturb_gene_ids_list = list(perturb_gene_ids['gene_id'])

gene_id_to_name = dict(zip(lookup_df['gene_id'], lookup_df['gene_name']))

df = pd.read_excel(vst_path)
vst_df = df.set_index("Sample")

gene_names = [gene_id_to_name.get(gid, gid) for gid in vst_df.columns]
# 308 x 84
X_full = vst_df.values
def get_intervened_gene(sample_name):
    tokens = str(sample_name).upper().split("_")
    return tokens[2] if len(tokens) >= 3 else None
#308
intervened_genes = [get_intervened_gene(s) for s in vst_df.index]

print(X_full.shape)
print(len(perturb_gene_ids_list))

edges_with_weight = []
for target_idx, target_gene in enumerate(vst_df.columns):
    target_gene_name = gene_id_to_name.get(target_gene, target_gene)


    selector = [col in perturb_gene_ids_list and col != target_gene for col in vst_df.columns]

    valid_rows = [
        i for i, g in enumerate(intervened_genes)
        if g != target_gene_name
    ]
    X = X_full[valid_rows][:, selector]
    y = X_full[valid_rows][:, target_idx]
    X_scaled = StandardScaler().fit_transform(X)

    model = Lasso(alpha=0.05, random_state=42)
    model.fit(X_scaled, y)
    coefs = model.coef_

    parent_names = np.array(gene_names)[selector]
    for parent, weight in zip(parent_names, coefs):
        if weight != 0:
            edges_with_weight.append((parent, gene_names[target_idx], weight))

# full edge
edges_all_df = pd.DataFrame(edges_with_weight, columns=["Source", "Target", "Weight"])
edges_all_df.to_csv("lasso_causal_edges_all.csv", index=False)

# 84x84
perturbed_gene_names = [gene_id_to_name.get(gid, gid) for gid in perturb_gene_ids_list]
edges_84x84_df = edges_all_df[
    edges_all_df['Source'].isin(perturbed_gene_names) &
    edges_all_df['Target'].isin(perturbed_gene_names)
]
edges_84x84_df.to_csv("lasso_causal_edges_84x84.csv", index=False)