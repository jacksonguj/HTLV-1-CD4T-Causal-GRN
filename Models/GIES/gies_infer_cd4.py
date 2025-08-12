import numpy as np
import pandas as pd
from causalscbench.models.gies import GIES
from causalscbench.models.training_regimes import TrainingRegime

# Load dataset
data = np.load("dataset_cd4_filtered.npz", allow_pickle=True)
expression_matrix = data["expression_matrix"]
interventions = data["interventions"]
gene_names = data["var_names"]

# GIES inference
model = GIES()
edges = model(
    expression_matrix=expression_matrix,
    interventions=interventions.tolist(),
    gene_names=gene_names.tolist(),
    training_regime=TrainingRegime.Interventional,
    seed=0
)

# Save results
edges_df = pd.DataFrame(edges, columns=["source", "target"])
edges_df.to_csv("gies_cd4_edges.csv", index=False)
print("Inferred edges saved to gies_cd4_edges.csv")
