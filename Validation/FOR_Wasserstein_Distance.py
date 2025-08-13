
import rpy2.robjects as robjects
from rpy2.robjects import r
import numpy as np
import pandas as pd

readRDS = robjects.r['readRDS']
cd4_expression = readRDS("cd4_expression.rds")
gene_ids = readRDS("gene_ids.rds")
ko_genes_cleaned = readRDS("ko_genes_cleaned.rds")

expr_df = pd.DataFrame(np.array(cd4_expression))
expr_matrix = expr_df.values
var_names = np.array(gene_ids)
interventions = np.array(ko_genes_cleaned)

np.savez("dataset_cd4_filtered.npz",
         expression_matrix=expr_matrix,
         var_names=var_names,
         interventions=interventions)

import rpy2.robjects as robjects
from rpy2.robjects import r
import numpy as np
import pandas as pd

readRDS = robjects.r['readRDS']
cd4_expression = readRDS("cd4_expression.rds")
gene_ids = readRDS("gene_ids.rds")
ko_genes_cleaned = readRDS("ko_genes_cleaned.rds")

expr_df = pd.DataFrame(np.array(cd4_expression)).T
expr_matrix = expr_df.values
var_names = np.array(gene_ids)
interventions = np.array(ko_genes_cleaned)

np.savez("dataset_cd4_filtered.npz",
         expression_matrix=expr_matrix,
         var_names=var_names,
         interventions=interventions)

data = np.load("dataset_cd4_filtered.npz", allow_pickle=True)
print("expression_matrix shape:", data["expression_matrix"].shape)
print("len(interventions):", len(data["interventions"]))

import numpy as np
import pandas as pd
import scipy
import random
from typing import List, Tuple, Dict

class Evaluator(object):
    def __init__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        p_value_threshold=0.05,
    ) -> None:
        self.gene_to_index = dict(zip(gene_names, range(len(gene_names))))
        self.gene_to_interventions = dict()
        for i, intervention in enumerate(interventions):
            self.gene_to_interventions.setdefault(intervention, []).append(i)
        self.expression_matrix = expression_matrix
        self.p_value_threshold = p_value_threshold
        self.gene_names = gene_names

    def get_observational(self, child: str) -> np.array:
        return self.get_interventional(child, "non-targeting")

    def get_interventional(self, child: str, parent: str) -> np.array:
        return self.expression_matrix[
            self.gene_to_interventions[parent], self.gene_to_index[child]
        ]

    def evaluate_network(self, network: List[Tuple], max_path_length = 3, check_false_omission_rate=False, omission_estimation_size=0, seed=0) -> Dict:
        network_as_dict = {}
        for a, b in network:
            network_as_dict.setdefault(a, set()).add(b)
        true_positive, false_positive, wasserstein_distances = self._evaluate_network(
            network_as_dict
        )

        all_connected_network = {**network_as_dict}
        all_path_results = []
        if max_path_length == -1:
            max_path_length = len(self.gene_names)
        for _ in range(max_path_length - 1):
            single_step_deeper_all_connected_network = {
                v: n.union(
                    *[
                        all_connected_network[nn]
                        for nn in n
                        if nn in all_connected_network
                    ]
                )
                for v, n in all_connected_network.items()
            }
            single_step_deeper_all_connected_network = {v: c - {v} for v, c in single_step_deeper_all_connected_network.items()}
            if single_step_deeper_all_connected_network == all_connected_network:
                break
            new_edges = {
                key: single_step_deeper_all_connected_network[key] - all_connected_network[key]
                for key in single_step_deeper_all_connected_network
            }
            (
                true_positive_connected,
                false_positive_connected,
                wasserstein_distances_connected,
            ) = self._evaluate_network(new_edges)
            all_path_results.append({
                "true_positives": true_positive_connected,
                "false_positives": false_positive_connected,
                "wasserstein_distance": {
                    "mean": np.mean(wasserstein_distances_connected)
                },
            })
            all_connected_network = single_step_deeper_all_connected_network

        if check_false_omission_rate:
            edges = set()
            random.seed(seed)
            while len(edges) < omission_estimation_size:
                pair = random.sample(range(len(self.gene_names)), 2)
                edge = self.gene_names[pair[0]], self.gene_names[pair[1]]
                if edge[0] in all_connected_network and edge[1] in all_connected_network[edge[0]]:
                    continue
                edges.add(edge)
            network_as_dict = {}
            for a, b in edges:
                network_as_dict.setdefault(a, set()).add(b)
            res_random = self._evaluate_network(network_as_dict)
            false_omission_rate = res_random[0] / omission_estimation_size
            negative_mean_wasserstein = np.mean(res_random[2])
        else:
            false_omission_rate = -1
            negative_mean_wasserstein = -1

        return {
            "output_graph": {
                "true_positives": true_positive,
                "false_positives": false_positive,
                "wasserstein_distance": {"mean": np.mean(wasserstein_distances)},
            },
            "all_path_results": all_path_results,
            "false_omission_rate": false_omission_rate,
            "negative_mean_wasserstein": negative_mean_wasserstein
        }

    def _evaluate_network(self, network_as_dict):
        true_positive = 0
        false_positive = 0
        wasserstein_distances = []
        for parent in network_as_dict.keys():
            children = network_as_dict[parent]
            for child in children:
                try:
                    observational_samples = self.get_observational(child)
                    interventional_samples = self.get_interventional(child, parent)
                    ranksum_result = scipy.stats.mannwhitneyu(
                        observational_samples, interventional_samples
                    )
                    wasserstein_distance = scipy.stats.wasserstein_distance(
                        observational_samples, interventional_samples
                    )
                    wasserstein_distances.append(wasserstein_distance)
                    p_value = ranksum_result[1]
                    if p_value < self.p_value_threshold:
                        true_positive += 1
                    else:
                        false_positive += 1
                except:
                    continue
        return true_positive, false_positive, wasserstein_distances

npz = np.load("dataset_cd4_filtered.npz", allow_pickle=True)
expression_matrix = npz["expression_matrix"]
gene_names = npz["var_names"].tolist()
interventions = npz["interventions"].tolist()

edge_df = pd.read_csv("dcdi_dsf_edges_84x13000.csv")
network = list(zip(edge_df["Source"], edge_df["Target"]))

evaluator = Evaluator(expression_matrix, interventions, gene_names)
#result = evaluator.evaluate_network(network)
result = evaluator.evaluate_network(
    network,
    max_path_length=3,
    check_false_omission_rate=True,
    omission_estimation_size=1000,
    seed=0
)

import pprint
pprint.pprint(result)

print("(interventions):", len(interventions))
print("(expression_matrix):", expression_matrix.shape[0])