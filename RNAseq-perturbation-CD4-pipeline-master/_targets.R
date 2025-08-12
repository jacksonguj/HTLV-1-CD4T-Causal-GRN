# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

# Load packages required to define the pipeline:
library(targets)
# library(tarchetypes) # Load other packages as needed. # nolint

# Set target options:
tar_option_set(
  packages = c(
    "tibble",
    "magrittr",
    "dplyr",
    "purrr",
    "DESeq2",
    "stringr",
    "rtracklayer",
    "vroom",
    "arrow",
    "ggplot2",
    "patchwork",
    "mashr",
    "logger",
    "enrichR",
    "ggridges",
    "ggraph"
  ), # packages that your targets need to run
  format = "qs" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
# options(clustermq.scheduler = "slurm")
# options(clustermq.template = "clustermq.tmpl")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# Load the R scripts with your custom functions:
source("R/functions.R")
source("R/plot_network.R")
source("R/enrichment_analyses.R")
source("R/mashr_analysis.R")
source("R/proliferation.R")
source("R/joint_downstream_model.R")
source("R/validate_network.R")
source("R/ABC_enrichment.R")
source("R/CHiP_enrichment.R")
source("R/pathway_plot.R")
source("R/annotate_GWAS_loci.R")
source("R/compare_to_yazar.R")
source("R/perez_analysis.R")
source("R/SLE_gene_signatures.R")

# Replace the target list below with your own:
list(
    tar_target(
        name = rna,
        command =  read_rna(rna_seq_location()),
        format = "parquet"
    ),
    tar_target(
        name = txdb, # library(rtracklayer)
        command = get_txdb(gencode_location()),
        format = "parquet"
    ),
    tar_target(
        name = meta,
        command = parse_metadata(rna, vroom::vroom(master_metadata())),
        format = "parquet"
    ),
    tar_target(
        name = counts,
        command = get_counts(rna)
    ),
    tar_target(
        name = dds, # DESeqDataSet, library(DESeq2)
        command = make_dds(counts, meta)
    ),
    tar_target(
        name = filtered_dds, # library(logger)
        command = filter_low_counts(dds)
    ),
    tar_target(
        name = pca,
        command = apply_pca(SummarizedExperiment::assay(DESeq2::vst(filtered_dds, blind = TRUE)))
    ),
    tar_target(
        name = dds_with_pcs,
        command = modify_dds(filtered_dds, pca)
    ),
    tar_target(
        name = diffeq_no_pcs,
        command = run_diffeq(filtered_dds)
    ),
    tar_target(
        name = diffeq_with_pcs,
        command = run_diffeq(dds_with_pcs)
    ),
    tar_target(
        name = gene_lookup,
        command = get_readout_gene_lookup(filtered_dds, txdb),
        format = "parquet"
    ),
    tar_target(
        name = plot_pca_,
        command = plot_pca(pca, meta)
    ),
    tar_target(
        name = results_with_pcs, # library(stringr, purrr)
        command = parse_results(diffeq_with_pcs)
    ),
    tar_target(
        name = results_no_pcs,
        command = parse_results(diffeq_no_pcs)
    ),
    tar_target(
        name = compare_dge_,
        command = compare_dge(results_with_pcs, results_no_pcs, txdb),
        format = "parquet"
    ),
    tar_target(
        name = plot_compare_dge_,
        command = plot_compare_dge(compare_dge_),
        format = "file"
    ),
    tar_target(
        name = mashr,
        command = apply_mashr(results_with_pcs, txdb)
    ),
    tar_target(
        name = mashr_no_pcs,
        command = apply_mashr(results_no_pcs, txdb)
    ),
    tar_target(
        name = joint_downstream_model,
        command = read_joint_downstream_model(txdb)
    ),
    tar_target(
        name = pairwise,
        command = extract_pairwise_effects(mashr)
    ),
    tar_target(
        name = pairwise_joint_downstream,
        command = extract_pairwise_effects(joint_downstream_model)
    ),
    tar_target(
        name = plot_pairwise,
        command = plot_pairwise_effects(pairwise, meta)
    ),
    tar_target(
        name = plot_pairwise_joint_downstream,
        command = plot_pairwise_effects(pairwise_joint_downstream, meta, tag = "joint_downstream")
    ),
    tar_target(
        name = plot_covariance_pis_,
        command = plot_covariance_pis(mashr)
    ),
    tar_target(
        name = causal_network,
        command = read_direct_effects(direct_effects_location())
    ),
    tar_target(
        name = permuted_networks,
        command = permute_graph_and_tabulate_clustering(causal_network, meta, set_causal_threshold()),
        packages = c(
            "dplyr",
            "tidygraph",
            "magrittr"
        )
    ),
    tar_target(
        name = plot_permuted_networks,
        command = compare_network_to_permutations(causal_network, permuted_networks, meta),
        packages = c(
            "dplyr",
            "ggplot2",
            "magrittr"
        )

    ),
    tar_target(
        name = write_out_total_covar,
        command = compute_covariance_reorder(causal_network, txdb)
    ),
    tar_target(
        name = length_2_cycle_genes,
        command = find_cycles(causal_network, meta, 2)
    ),
    tar_target(
        name = length_3_cycle_genes,
        command = find_cycles(causal_network, meta, 3)
    ),
    tar_target(
        name = compute_centrality_,
        command = compute_centrality(causal_network, meta)
    ),
    tar_target(
        name = plot_causal_network,
        command = plot_network(
            causal_network, 
            meta, 
            threshold = 0.025, 
            tag = "causal",
            layout = "stress",
            niter = 700, 
            bbox = 20,
            scale_node_size_by_degree = TRUE,
            width = 13.5,
            height = 10,
            label_size = 6.3,
            label_nodes = TRUE
        ),
        format = "file"
    ),
    tar_target(
        name = plot_causal_network_IEI_emphasize,
        command = plot_network(
            causal_network, 
            meta, 
            threshold = 0.025, 
            tag = "causal_emphasize_IEI",
            layout = "stress",
            niter = 700, 
            bbox = 20,
            scale_node_size_by_degree = TRUE,
            width = 13.5,
            height = 10,
            label_size = 12.0,
            label_nodes = TRUE,
            filter_IEI = TRUE,
            edge_width = 1.5
        ),
        format = "file"
    ),
    tar_target(
        name = plot_causal_network_no_label,
        command = plot_network(
            causal_network, 
            meta, 
            threshold = 0.025, 
            tag = "causal_no_labels",
            layout = "stress",
            niter = 700, 
            bbox = 20,
            scale_node_size_by_degree = TRUE,
            width = 13.5,
            height = 10,
            label_size = 6.3,
            label_nodes = FALSE
        ),
        format = "file"
    ),
    tar_target(
        name = plot_causal_network_sub_network_stat1_,
        command = plot_network(
            causal_network %>% dplyr::filter(row == "STAT1" | col == "STAT1"), 
            meta, 
            threshold = 0.025,
            tag = "STAT1",
            edges = "curved",
            layout = "auto",
            scale_node_size_by_degree = FALSE,
            width = 7,
            height = 5,
            label_size = 4
        ),
        format = "file"
    ),
    tar_target(
        name = plot_causal_network_sub_network_med12_,
        command = plot_network(
            causal_network %>% dplyr::filter(row == "MED12" | col == "MED12"), 
            meta, 
            threshold = 0.025,
            tag = "MED12",
            edges = "straight",
            layout = "stress",
            scale_node_size_by_degree = FALSE,
            width = 7,
            height = 5,
            label_size = 4
        ),
        format = "file"
    ),
    # tar_target(
    #     name = plot_causal_network_dge,
    #     command = plot_network(causal_network, meta, threshold = NULL, results = results_with_pcs, txdb),
    #     format = "file"
    # ),
    tar_target(
        name = network_degree,
        command = compute_degree(causal_network)
    ),
    tar_target(
        name = stratify_network_degree_,
        command = stratify_degree_by_gene_group(network_degree, meta)
    ),
    tar_target(
        name = total_effects,
        command = read_total_effects()
    ),
    tar_target(
        name = compare_total_effects_,
        command = compare_total_effects(total_effects, results_with_pcs, txdb)
    ),
    tar_target(
        name = compare_diffeq_to_causal_,
        command = compare_direct_diffeq_effects(causal_network, results_with_pcs, txdb, expression)
    ),
    tar_target(
        name = plot_direct_indirect_,
        command = plot_direct_indirect(causal_network, total_effects, results_with_pcs, txdb, length_2_cycle_genes)
    ),
    tar_target(
        name = plot_indegree_outdegree_scatter_,
        command = plot_indegree_outdegree_scatter(compute_centrality_)
    ),
    tar_target(
        name = plot_causal_scatter,
        command = compare_pairwise_to_direct_effects(pairwise, causal_network)
    ),
    tar_target(
        name = plot_causal_scatter_joint_downstream,
        command = compare_pairwise_to_direct_effects(pairwise_joint_downstream, causal_network, tag = "joint_downstream")
    ),
    tar_target(
        name = plot_causal_scatter_distance,
        command = compare_pairwise_to_distance(pairwise, causal_network, meta)
    ),
    tar_target(
        name = plot_causal_scatter_distance_joint_downstream,
        command = compare_pairwise_to_distance(pairwise_joint_downstream, causal_network, meta, tag = "joint_downstream")
    ),
    tar_target(
        name = plot_degree_scatter,
        command = compare_degree_estimates(results_with_pcs, causal_network)
    ),
    tar_target(
        name = ko_specific_effects,
        command = extract_ko_specific_effects(mashr, txdb)
    ),
    tar_target(
        name = ko_specific_effects_no_pcs,
        command = extract_ko_specific_effects(mashr_no_pcs, txdb)
    ),
    tar_target(
        name = ko_specific_effects_joint_downstream,
        command = extract_ko_specific_effects(joint_downstream_model, txdb)
    ),
    tar_target(
        name = plot_top_downstream_indegree_,
        command = plot_top_downstream_indegree(joint_downstream_model, expression, meta)
    ),
    tar_target(
        name = downstream_indegree,
        command = compute_downstream_indegree(mashr, txdb)
    ),
    tar_target(
        name = downstream_indegree_by_group,
        command = compute_downstream_indegree_by_group(mashr, txdb, meta, type = "lfsr", pm_threshold = 5e-3)
    ),
    tar_target(
        name = downstream_indegree_by_group_joint_downstream,
        command = compute_downstream_indegree_by_group(joint_downstream_model, txdb, meta, type = "lfsr", lfsr_threshold = 5e-3)
    ),
    tar_target(
        name = constraint,
        command = vroom::vroom(constraint_metrics_location(), col_select = c(gene, gene_id, pLI)) %>%
            dplyr::rename(gene_name = gene) %>%
            dplyr::left_join(
                vroom::vroom(tj_constraint_metrics_location()) %>%
                    dplyr::select(
                        gene_id = ensg,
                        tj_constraint = post_mean
                    )
            ) %>%
            dplyr::select(-gene_id)
    ),
    tar_target(
        name = proliferation,
        command = read_proliferation()
    ),
    tar_target(
        name = plot_proliferation_scatter_,
        command = plot_proliferation(meta, proliferation)
    ),
    tar_target(
        name = cytokine_hits,
        command = read_schmidt_cytokine_hits()
    ),
    tar_target(
        name = joint_downstream_IFNg_proliferation,
        command = plot_cytokine_proliferation_effect(proliferation, cytokine_hits, joint_downstream_model, meta)
    ),
    tar_target(
        name = enrichment_,
        command = enrichment(ko_specific_effects)
    ),
    tar_target(
        name = enrichment_joint_downstream_,
        command = enrichment(ko_specific_effects_joint_downstream)
    ),
    tar_target(
        name = plot_ko_specific_effects_,
        command = plot_ko_specific_effects(ko_specific_effects, meta, tag = "regressed_pcs")
    ),
    tar_target(
        name = plot_ko_specific_effects_no_pcs_,
        command = plot_ko_specific_effects(ko_specific_effects_no_pcs, meta, tag = "no_pcs")
    ),
    tar_target(
        name = plot_ko_specific_effects_joint_downstream_,
        command = plot_ko_specific_effects(ko_specific_effects_joint_downstream, meta, tag = "joint_downstream")
    ),
    tar_target(
        name = plot_ko_specific_effects_proliferation_,
        command = plot_proliferation_ko_specific(ko_specific_effects, proliferation, meta)
    ),
    tar_target(
        name = plot_ko_specific_effects_proliferation_joint_downstream,
        command = plot_proliferation_ko_specific(ko_specific_effects_joint_downstream, proliferation, meta, tag = "joint_downstream")
    ),
    tar_target(
        name = plot_enrichments_,
        command = plot_enrichments(enrichment_)
    ),
    tar_target(
        name = plot_enrichments_joint_downstream,
        command = plot_enrichments(enrichment_joint_downstream_, tag = "joint_downstream")
    ),
    tar_target(
        name = downstream_gene_enrichment,
        command = disease_enrichment_of_coregulation(downstream_indegree, diffeq_no_pcs, gene_lookup)
    ),
    tar_target(
        name = plot_strata_,
        command = plot_strata(downstream_gene_enrichment)
    ),
    tar_target(
        name = plot_downstream_cluster_,
        command = downstream_cluster(
            mashr, 
            txdb, 
            row_cluster_k = 8,
            col_cluster_k = 8,
            min_overlap_prop = 0,
            min_overlap = 0,
            enrichment_function = "enrichr", 
            tag = "diffeq"
        )
    ),
    tar_target(
        name = plot_downstream_cluster_joint_downstream_,
        command = downstream_cluster(
            joint_downstream_model, 
            txdb, 
            row_cluster_k = 7,
            col_cluster_k = 4,
            lower_limit_n_downstream = 100,
            min_overlap_prop = 0,
            min_overlap = 0,
            enrichment_function = "enrichr", 
            tag = "joint_downstream"
        )
    ),
    tar_target(
        name = plot_downstream_cluster_predefined_,
        command = downstream_cluster(
            mashr,
            txdb,
            enrichment_function = "predefined",
            gene_set = iei_gene_list(),
            gene_names = iei_genes(),
            row_cluster_k = 5,
            min_overlap = 1,
            min_overlap_prop = .03,
            gene_group_label = "predefined",
            tag = "diffeq"
        )
    ),
    tar_target(
        name = plot_downstream_cluster_predefined_joint_downstream_,
        command = downstream_cluster(
            joint_downstream_model,
            txdb,
            enrichment_function = "predefined",
            gene_set = iei_gene_list(),
            gene_names = iei_genes(),
            row_cluster_k = 5,
            min_overlap = 1,
            min_overlap_prop = .03,
            gene_group_label = "predefined",
            tag = "joint_downstream"
        )
    ),
    tar_target(
        name = plot_downstream_cluster_iei_,
        command = downstream_cluster(
                    mashr,
                    txdb,
                    gene_names = iei_genes(),
                    gene_group_label = "iei",
                    lower_limit_n_downstream = 10,
                    row_cluster_k = 8,
                    col_cluster_k = 8,
                    # min_overlap = 5,
                    # min_overlap_prop = .18,
                    enrichment_function = "enrichr",
                    tag = "diffeq",
                    database = "GO_Biological_Process_2021"
                )
    ),
    tar_target(
        name = plot_downstream_cluster_iei_joint_downstream_,
        command = downstream_cluster(
                    joint_downstream_model,
                    txdb,
                    gene_names = iei_genes(),
                    gene_group_label = "iei",
                    lower_limit_n_downstream = 20,
                    row_cluster_k = 6,
                    col_cluster_k = 4,
                    # min_overlap = 5,
                    # min_overlap_prop = .18,
                    enrichment_function = "enrichr",
                    tag = "joint_downstream",
                    database = "GO_Biological_Process_2021"
                )
    ),
    tar_target(
        name = plot_downstream_cluster_pics_,
        command = downstream_cluster(
                    mashr,
                    txdb,
                    gene_names = pics_ai_genes(),
                    gene_group_label = "pics",
                    row_cluster_k = 8,
                    col_cluster_k = 8,
                    # min_overlap = 5,
                    # min_overlap_prop = .18,
                    enrichment_function = "enrichr",
                    tag = "diffeq",
                    "GO_Biological_Process_2021"
                )
    ),
    tar_target(
        name = plot_downstream_cluster_pics_joint_downstream_,
        command = downstream_cluster(
                    joint_downstream_model,
                    txdb,
                    gene_names = pics_ai_genes(),
                    gene_group_label = "pics",
                    row_cluster_k = 7,
                    col_cluster_k = 7,
                    # min_overlap = 5,
                    # min_overlap_prop = .18,
                    enrichment_function = "enrichr",
                    tag = "joint_downstream",
                    "GO_Biological_Process_2021"
                )
    ),
    tar_target(
        name = gene_mask,
        command = create_gene_mask(gene_lookup, meta),
        format = "feather"
    ),
    tar_target(
        name = write_gene_mask_,
        command = write_gene_mask(gene_mask),
        format = "file"
    ),
    # tar_target(
    #     name = read_yazar_,
    #     command = read_yazar_rans_eqtls()
    # ),
    tar_target(
        name = cis_yazar,
        command = read_yazar_cis_eqtls(meta)
    ),
    tar_target(
        name = trans_yazar,
        command = read_yazar_trans_eqtls(cis_yazar)
    ),
    tar_target(
        name = yazar_comparison,
        command = compare_to_yazar(cis_yazar, trans_yazar, constraint, meta)
    ),
    tar_target(
        name = tcell_subtype_specific_genes,
        command = read_dice(),
        format = "feather"
    ),
    tar_target(
        name = blood_specific_genes,
        command = identify_blood_specific_genes()
    ),
    tar_target(
        name = read_eqtlgen_,
        command = read_eqtlgen_trans_eqtls()
    ),
    tar_target(
        name = expression,
        command = extract_expression_by_gene(results_with_pcs, txdb),
    ),
    tar_target(
        name = forest_plot_downstream_indegree_pLI_,
        command = forest_plot_downstream_indegree(
            downstream_indegree_by_group,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint_measure = "pLI",
            tag = "diffeq_lfsr_pLI"
        )
    ),
    tar_target(
        name = forest_plot_downstream_indegree_TJ_,
        command = forest_plot_downstream_indegree(
            downstream_indegree_by_group,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint_measure = "tj_constraint",
            tag = "diffeq_lfsr_TJ"
        )
    ),
    tar_target(
        name = forest_plot_downstream_indegree_joint_downstream_pLI_,
        command = forest_plot_downstream_indegree(
            downstream_indegree_by_group_joint_downstream,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint_measure = "pLI",
            tag = "joint_downstream_lfsr_pLI"
        )
    ),
    tar_target(
        name = forest_plot_downstream_indegree_joint_downstream_TJ_,
        command = forest_plot_downstream_indegree(
            downstream_indegree_by_group_joint_downstream,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint_measure = "tj_constraint",
            tag = "joint_downstream_lfsr_TJ"
        )
    ),
    tar_target(
        name = forest_plot_downstream_indegree_cell_type_specific_joint_downstream_,
        command = forest_plot_downstream_indegree_cell_type_specific(
            downstream_indegree_by_group_joint_downstream,
            expression,
            meta,
            TH1_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "TH1 cells"),
            TH2_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "TH2 cells"),
            TH17_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "TH17 cells"),
            naive_TREG_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "Naïve TREG cells"),
            naive_CD4_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "Naïve CD4+ T cells"),
            memory_TREG_genes = pull_cell_type_specific_genes(tcell_subtype_specific_genes, "Memory TREG cells"),
            tag = "joint_downstream_lfsr"
        )
    ),
    tar_target(
        name = forest_plot_causal_centrality_pLI_,
        command = forest_plot_causal_centrality(
            compute_centrality_,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            constraint_measure = "pLI",
            tag = "pLI"
        )
    ),
    tar_target(
        name = forest_plot_causal_centrality_TJ_,
        command = forest_plot_causal_centrality(
            compute_centrality_,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name,
            constraint_measure = "tj_constraint",
            tag = "TJ"
        )
    ),
    # total graph analyses
    tar_target(
        name = total_graph,
        command = create_total_tidy_graph(
            causal_network,
            joint_downstream_model,
            expression,
            meta,
            constraint,
            iei_genes = iei_genes(),
            gwas_genes = pics_ai_genes(),
            trans_egenes = unique(read_eqtlgen_)$gene_name
        )
    ),
    tar_target(
        name = scatter_upstream_constraint_,
        command = constraint_upstream_scatter(total_graph)
    ),
    # tar_target(
    #     name = plot_total_graph_gwas,
    #     command = plot_total_graph(total_graph, color = "is_gwas", color_label = "AI GWAS gene", tag = "gwas")
    # ),
    # tar_target(
    #     name = plot_total_graph_iei,
    #     command = plot_total_graph(total_graph, color = "is_iei", color_label = "IEI gene", tag = "iei")
    # ),
    # tar_target(
    #     name = plot_total_graph_eqlten,
    #     command = plot_total_graph(total_graph, color = "is_trans_egene", color_label = "eQTLgen trans-eGene", tag = "eqtlgen")
    # ),
    tar_target(
        name = hbase,
        command = read_hbase(meta)
    ),
    tar_target(
        name = stringdb,
        command = read_stringdb(meta)
    ),
    tar_target(
        name = determine_enrichment_hbase,
        command = iterate_hbase_thresholding(meta, hbase, causal_network)
    ),
    tar_target(
        name = ABC_GRN,
        command = read_ABC_GRN(meta)
    ),
    tar_target(
        name = DAC,
        command = read_DAC()
    ),
    tar_target(
        name = ChIP,
        command = read_ChIP()
    ),
    tar_target(
        name = ChIP_ABC_GRN,
        command = merge_ABC_ChIP_GRN(ABC_GRN, ChIP)
    ),
    tar_target(
        name = DAC_ABC_GRN,
        command = merge_DAC_ABC_GRN(ABC_GRN, DAC)
    ),
    tar_target(
        name = determine_enrichment_DAC_ABC_GRN,
        command = iterate_ABC_GRN_thresholding(meta, DAC_ABC_GRN, causal_network)
    ),
    tar_target(
        name = determine_enrichment_ChIP_ABC_GRN,
        command = iterate_ChIP_thresholding(meta, ChIP_ABC_GRN, causal_network)
    ),
    tar_target(
        name = matrix_factorization,
        command = read_matrix_factorization(txdb)
    ),
    tar_target(
        name = matrix_factorization_enrichments,
        command = enrich_loadings(matrix_factorization, txdb)
    ),
    tar_target(
        name = program_labels,
        command = process_loading_enrichments(matrix_factorization_enrichments)
    ),
    tar_target(
        name= plot_loadings_,
        command = plot_loadings(matrix_factorization, program_labels, txdb, meta)
    ),
    tar_target(
        name = pathway_plot_,
        command = plot_pathway(joint_downstream_model, txdb, "DR1", "R-HSA-202424")
    ),
    tar_target(
        name = heatmap_cluster_annotations, #manually derived,
        command = read_cluster_membership()
    ),
    tar_target(
        name = RA_GWAS_loci,
        command = annotate_RA_GWAS_loci(joint_downstream_model, heatmap_cluster_annotations, meta)
    ),
    tar_target(
        name = MS_GWAS_loci,
        command = annotate_MS_GWAS_loci(joint_downstream_model, heatmap_cluster_annotations, meta)
    ),
    tar_target(
        name = immune_GWAS_loci,
        command = read_in_immune_GWAS()
    ),
    tar_target(
        name = annotate_pan_immune_GWAS,
        command = pan_immune_GWAS_enrichment(
            immune_GWAS_loci, 
            heatmap_cluster_annotations, 
            joint_downstream_model,
            txdb,
            meta
        )
    ),
    tar_target(
        name = subset_causal_to_RA,
        command= subset_causal_network_to_RA_gwas(causal_network, meta)
    ),
    tar_target(
        name = RA_drug_upstream,
        command = plot_drug_network(
            joint_downstream_model,
            drug_targets = read_RA_drugs(),
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint,
            meta,
            tag = "RA"
        )
    ),
    tar_target(
        name = MS_drug_upstream,
        command = plot_drug_network(
            joint_downstream_model,
            drug_targets = read_MS_drugs(),
            blood_specific_genes = convert_ensembl_to_symbol(blood_specific_genes, remove_ensembl_version_txdb(txdb)),
            constraint,
            meta,
            tag = "MS"
        )
    ),
    tar_target(
        name = cluster_4_gwas_subnetwork_,
        command = create_module_gwas_graph(causal_network, heatmap_cluster_annotations, annotate_pan_immune_GWAS, meta)
        # command = create_module_gwas_graph(causal_network, heatmap_cluster_annotations, joint_downstream_model, ai_genes = pics_ai_genes(), meta)
    ),
    tar_target(
        name = cluster_4_JAK_STAT_subnetwork_,
        command = create_module_pathway_graph(causal_network, joint_downstream_model, heatmap_cluster_annotations, "04630", meta)
    ),
    tar_target(
        name = cluster_4_Th17_subnetwork_,
        command = create_module_th17_manual_graph(causal_network, joint_downstream_model, heatmap_cluster_annotations, meta)
    ),
    tar_target(
        name = cluster_2A_cell_cycle_subnetwork_,
        command = create_module_cell_cycle_manual_graph(causal_network, joint_downstream_model, heatmap_cluster_annotations, meta, lfsr_threshold = 0.05)
    ),
    tar_target(
        name = effector_genes,
        command = read_effectorness_genes()
    ),
    tar_target(
        name = effector_score,
        command = compute_effector_score(joint_downstream_model, effector_genes, cytokine_hits, heatmap_cluster_annotations, meta)
    ),
    tar_target(
        name = pathfindr_results,
        command = read_pathfindR_results()
    ),
    # tar_target(
    #     name = create_pathfinder_graph_,
    #     command = create_pathfinder_graph(causal_network, pathfindr_results, heatmap_cluster_annotations, meta)
    # ),
    tar_target(
        name = module_genes,
        command = compute_modules_with_downstream(
                    joint_downstream_model,
                    heatmap_cluster_annotations,
                    expression
        )
    ),
    tar_target(
        name = lupus_cells,
        command =  read_lupus_cells()
    ),
    # tar_target(
    #     name = regress_signatures_lupus,
    #     command = regress_module_signatures(
    #         lupus_cells,
    #         module_genes,
    #         covars = lupus_covars(),
    #         main_effect = "disease"
    #     )
    # ),
    # tar_target(
    #     name = regress_signatures_lupus_state,
    #     command = regress_module_signatures(
    #         lupus_cells,
    #         module_genes,
    #         covars = lupus_state_covars(),
    #         main_effect = "disease_state"
    #     )
    # ),
    # tar_target(
    #     name = regress_signatures_cell_cycle,
    #     command = regress_module_signatures(
    #         lupus_cells,
    #         module_genes,
    #         covars = cell_cycle_covars(),
    #         main_effect = "G2M_score"
    #     )
    # ),
    # tar_target(
    #     name = plot_lupus_signatures,
    #     command = plot_module_signatures(regress_signatures_lupus, tag = "SLE")
    # ),
    # tar_target(
    #     name = plot_cell_cycle_signatures,
    #     command = plot_module_signatures(regress_signatures_cell_cycle, tag = "G2M_score")
    # ),
    # tar_target(
    #     name = plot_lupus_state_signatures,
    #     command = plot_module_disease_state(regress_signatures_lupus_state, tag = "disease_state")
    # ),
    tar_target(
        name = nakano_SLE_activity_signatures,
        command = disease_activity_signatures_nakano()
    ),
    tar_target(
        name = nakano_SLE_state_signatures,
        command = disease_state_signatures_nakano()
    ),
    tar_target(
        name = nakano_SLE_activity_enrichments,
        command = compute_enrichments_nakano(
            nakano_SLE_activity_signatures,
            module_genes$only_positively_regulated,
            expression$gene_name
        )
    ),
    tar_target(
        name = nakano_SLE_state_enrichments,
        command = compute_enrichments_nakano(
            nakano_SLE_state_signatures,
            module_genes$only_positively_regulated,
            expression$gene_name
        )
    ),
    tar_target(
        name = forest_plot_nakano_disease_activity,
        command = plot_module_nakano(
            nakano_SLE_activity_enrichments
        )
    ),
    tar_target(
        name = forest_plot_nakano_disease_state,
        command = plot_module_nakano(
            nakano_SLE_state_enrichments,
            tag = "SLE_state"
        )
    ),
    tar_target(
        name = forest_plot_nakano_combined,
        command = plot_module_nakano_combined(
            nakano_SLE_state_enrichments,
            nakano_SLE_activity_enrichments
        )
    ),
    tar_target(
        name = write_out_experiment_,
        command = write_out_experiment(
                    meta,
                    diffeq_with_pcs,
                    pca,
                    mashr,
                    joint_downstream_model,
                    results_with_pcs,
                    results_no_pcs,
                    module_genes,
                    txdb
                ),
        format = "file"
    )
)

