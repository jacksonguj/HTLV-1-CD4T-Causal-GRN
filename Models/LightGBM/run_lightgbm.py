import numpy as np
import pandas as pd
from lightgbm_model import Custom
from causalscbench.models.training_regimes import TrainingRegime

# Load dataset
data = np.load("dataset_cd4_filtered.npz", allow_pickle=True)
expression_matrix = data["expression_matrix"]
interventions = data["interventions"]
gene_names = data["var_names"]

# Run Custom model
model = Custom()
edges_with_scores = model(
    expression_matrix=expression_matrix,
    interventions=interventions.tolist(),
    gene_names=gene_names.tolist(),
    training_regime=TrainingRegime.Interventional,
    seed=0
)

# Save edge list with score
df = pd.DataFrame(edges_with_scores, columns=["source", "target", "score"])
df.to_csv("custom_cd4_edges_with_score.csv", index=False)
print("Saved edges with score to custom_cd4_edges_with_score.csv")