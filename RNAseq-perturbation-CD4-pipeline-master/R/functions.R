library(magrittr)
library(dplyr)

# "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/counts/dedup_counts.txt"
rna_seq_location = function() {
    "data/counts/GSE271788_dedup_counts.txt"
}


gencode_location = function() {
    "data/gencode/gencode.v41.basic.annotation.gtf"
}

hbase_location = function() {
    "data/HumanBase/t_lymphocyte_top.gz"
}

stringdb_location = function (){
    "data/STRINGDB/9606.protein.links.v12.0.txt.gz"
}

read_hbase = function(meta) {
    # first two columsn are entrez gene ids, third column is a posterior probability

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 

    network = vroom::vroom(hbase_location(), 
            col_names = c("entrez_id_1", "entrez_id_2", "edge"),
            col_types = "iid" 
        ) %>%
        dtplyr::lazy_dt(key_by = c(entrez_id_1, entrez_id_2))

    logger::log_info("network has {nrow(network)} rows")
    db = org.Hs.eg.db::org.Hs.eg.db


    entrez_ids = AnnotationDbi::mapIds(
        db, 
        keys = targets, 
        column = "ENTREZID",
        keytype = "SYMBOL"
    ) %>%
        as.character %>%
        as.numeric %>%
        as.integer

    entrez_df = tibble::tibble(
        gene_name = targets,
        entrez_id = entrez_ids
    ) %>%
        dtplyr::lazy_dt(key_by = entrez_id)

    logger::log_info("now subsetting to KO'd genes")

    # 913 / 6972
    network = network %>%
        dplyr::filter((entrez_id_1 %in% entrez_ids) & (entrez_id_2 %in% entrez_ids))

    logger::log_info("done subsetting to KO'd genes")

    network = network %>%
        dplyr::inner_join(entrez_df, by = c("entrez_id_1" = "entrez_id")) %>%
        dplyr::rename(gene_name_1 = gene_name) %>%
        dplyr::inner_join(entrez_df, by = c("entrez_id_2" = "entrez_id")) %>%
        dplyr::rename(gene_name_2 = gene_name) %>%
        tibble::as_tibble(.)

    logger::log_info("network has {nrow(network)} rows")

    return(network)
}


read_stringdb = function(meta) {

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 

    # ensembl protein ids
    network = vroom::vroom(
        stringdb_location(), 
        col_types = c("cci"),
        col_names = c("protein_id_1", "protein_id_2", "score")
    ) 

    # vroom::problems(network)

    network = network %>%
        dplyr::mutate(
            across(c(protein_id_1, protein_id_2), ~stringr::str_remove(.x, "9606."))
        ) %>%
        dtplyr::lazy_dt(key_by = c(protein_id_1, protein_id_2))

    logger::log_info("network has {nrow(network)} rows")

    # db = EnsDb.Hsapiens.v86::EnsDb.Hsapiens.v86
    db = org.Hs.eg.db::org.Hs.eg.db

    # can't map 11/84 genes
    protein_ids = AnnotationDbi::mapIds(
        db, 
        keys = targets, 
        # column = "PROTEINID",
        column = "ENSEMBLPROT",
        keytype = "SYMBOL"
    ) %>%
        as.character 

    protein_df = tibble::tibble(
        gene_name = targets,
        protein_id = protein_ids
    ) %>%
        dtplyr::lazy_dt(key_by = protein_id)

    print(protein_df)

    logger::log_info("now subsetting to KO'd genes")

    # 10 / 6972
    # only 10 edges
    network = network %>%
        dplyr::filter((protein_id_1 %in% protein_ids) & (protein_id_2 %in% protein_ids))

    logger::log_info("done subsetting to KO'd genes")

    network = network %>%
        dplyr::inner_join(protein_df, by = c("protein_id_1" = "protein_id")) %>%
        dplyr::rename(gene_name_1 = gene_name) %>%
        dplyr::inner_join(protein_df, by = c("protein_id_2" = "protein_id")) %>%
        dplyr::rename(gene_name_2 = gene_name) %>%
        tibble::as_tibble(.)

    logger::log_info("network has {nrow(network)} rows")

    return(network)
}

read_rna = function(file = rna_seq_location()) {
    rna = data.table::fread(file) %>%
        tibble::as_tibble(.)

    nm = names(rna)
    names_replace = stringr::str_detect(nm, "output")
    nm[names_replace] = basename(nm[names_replace])
    nm = stringr::str_remove(nm, ".dedup.bam")

    rna %>%
        setNames(nm)
}

apply_pca = function(normalized_counts) {
    variances = matrixStats::rowVars(normalized_counts)
    n_most_variable = 500
    most_variable = head(order(rowVars(normalized_counts),decreasing=TRUE), n_most_variable)
    mat = scale(t(normalized_counts[most_variable, ]))
    return(prcomp(mat))
}


figure_dir = function() {
    "output/diffeq/figures"
}

txt_dir = function() {
    "output/diffeq/txt"
}

plot_pca = function(pca, meta) {
    scores = pca$x %>%
        as_tibble(rownames = "sample") %>%
        inner_join(meta, by = "sample")

    plot = ggplot(data = scores, aes(x = PC1, y = PC2, color = KO)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC1_PC2_intervention.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC3, y = PC4, color = KO)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC3_PC4_intervention.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC1, y = PC2, color = experiment_version)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC1_PC2_experiment_version.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC3, y = PC4, color = experiment_version)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC3_PC4_experiment_version.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC5, y = PC6, color = experiment_version)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC5_PC6_experiment_version.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC5, y = PC6, color = KO)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC5_PC6_intervention.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC7, y = PC8, color = KO)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC7_PC8_intervention.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC1, y = PC2, color = Donor)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC1_PC2_donor.pdf"), plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = scores, aes(x = PC3, y = PC4, color = Donor)) +
            geom_point() + 
            cowplot::theme_cowplot(font_size = 12)

    ggsave(file.path(figure_dir(), "PC3_PC4_donor.pdf"), plot, width = 6, height = 4, units = "in")
}

get_txdb = function(file = gencode_location()) {
    txdb = rtracklayer::import(file) %>%
        as.data.frame %>%
        as_tibble %>%
        select(gene_id, gene_name) %>%
        distinct
    txdb
}

master_metadata = function() {
    "data/metadata/sample_meta_data.tsv"
}

parse_metadata = function(rna, master = vroom::vroom(master_metadata())) {

    sample_meta = tibble(
        sample = rna %>%
            dplyr::select(starts_with("Donor")) %>%
            names
        ) %>%
        dplyr::inner_join(master, by = c("sample" = "Sample")) %>%
        dplyr::mutate(
            Donor = as.character(glue::glue("{Donor}_{experiment_version}")),
            KO = dplyr::case_when(
                is_control ~ "AAVS1",
                TRUE ~ KO
            )
        ) %>%
        dplyr::filter(!(sample %in% bad_sample_ids())) %>%
        dplyr::mutate(dplyr::across(c(Donor, KO, experiment_version), as.factor)) %>%
        dplyr::mutate(KO = relevel(KO, ref = "AAVS1"))

    return(sample_meta)
}

bad_sample_ids = function() {
    c(
        "Donor_1_AAVS1_1_2",
        "Donor_1_AAVS1_7_2",
        "Donor_4_AAVS1_6_1"
    )
}

# raw counts
get_counts = function(rna) {
    count_matrix = rna %>% 
            dplyr::select(starts_with("Donor")) %>%
            as.matrix

    # browser()
    # bad_ids = stringr::str_sub(bad_sample_ids(), end = nchar(bad_sample_ids()) - 2) # remove last two characters
    bad_ids = bad_sample_ids()
    idx = !(colnames(count_matrix) %in% bad_ids)

    rownames(count_matrix) = rna$Geneid
    count_matrix = count_matrix[, idx]

    logger::log_info(glue::glue("count_matrix has {ncol(count_matrix)} columns"))

    return(count_matrix)
}

make_dds = function(counts, meta) {

    dds = DESeqDataSetFromMatrix(
        countData = counts,
        colData = meta,
        design = ~Donor + KO

    )

    return(dds)
}

modify_dds = function(dds, pca) {

    scores = pca$x %>%
        as_tibble(rownames = "sample") 

    stopifnot(all(scores$sample == rownames(SummarizedExperiment::colData(dds))))

    # SummarizedExperiment::colData(dds) = dplyr::inner_join(
    SummarizedExperiment::colData(dds) = DataFrame(
        SummarizedExperiment::colData(dds), 
        scores %>% dplyr::select(PC1:PC10)
    )

    design(dds) = formula(~Donor + PC1 + PC2 + PC3 + PC4 + KO)

    return(dds)
}

run_diffeq = function(dds) {
    RhpcBLASctl::omp_set_num_threads(1L)
    RhpcBLASctl::blas_set_num_threads(1L)

    log_info("now running differential expression")

    dds_diffeq = DESeq(dds, parallel = TRUE, BPPARAM = BiocParallel::MulticoreParam(10L))

    log_info("done running differential expression")

    return(dds_diffeq)
}

compare_total_effects = function(total_effects, results, txdb) {

    results = results %>%
        dplyr::inner_join(txdb, by = c("gene" = "gene_id"))
    
    dfm = total_effects %>%
        dplyr::filter(row != col) %>%
        dplyr::inner_join(results, by = c("row" = "KO", "col" = "gene_name")) %>%
        dplyr::filter(padj < 5e-2) %>%
        dplyr::mutate(
              label = glue::glue("{row}%->%{col}"),
              parsed_label = map_chr(label, ~as.character(as.expression(.x)))
        )

    print(summary(lm(log2FoldChange ~ total_effect, data = dfm)))
    print(summary(lm(log2FoldChange ~ total_effect, data = dfm %>% dplyr::filter(baseMean > 500))))

    plot = ggplot(
        data = dfm, 
        aes(x = total_effect, y = log2FoldChange, color = baseMean, label = label)) +
        ggrepel::geom_text_repel(max.overlaps = 15, parse = TRUE) +
        geom_point() +
        scale_colour_viridis_c() + 
        geom_smooth(method = "lm") +
        labs(x = "linear total effect estimates", y = "DESeq2 log2FCs", colour = "baseline expression") +
        cowplot::theme_cowplot(font_size = 12)

    fname = file.path(figure_dir(), "compare_log2fcs_total_effect.pdf")
    ggsave(fname, plot, width = 8, height = 6, units = "in")
}

compare_direct_diffeq_effects = function(causal_network, results, txdb, expression, threshold = set_causal_threshold()) {

    results = results %>%
        dplyr::select(-baseMean) %>%
        dplyr::inner_join(txdb, by = c("gene" = "gene_id")) %>%
        dplyr::inner_join(
            expression %>%
                dplyr::select(KO = gene_name, KO_expression = expression),
            by = c("KO")
        ) %>%
        dplyr::inner_join(
            expression %>%
                dplyr::select(gene_name, child_expression = expression),
            by = c("gene_name")
        )
    
    dfm = causal_network %>%
        dplyr::filter(abs(estimate) > .env[["threshold"]]) %>%
        dplyr::inner_join(results, by = c("row" = "KO", "col" = "gene_name")) %>%
        dplyr::filter(padj < 5e-2) %>%
        dplyr::mutate(
              label = glue::glue("{row}%->%{col}"),
              parsed_label = map_chr(label, ~as.character(as.expression(.x)))
        )

    print(summary(lm(log2FoldChange ~ estimate, data = dfm)))
    print(summary(lm(log2FoldChange ~ estimate, data = dfm %>% dplyr::filter(KO_expression > 500))))
    print(summary(lm(log2FoldChange ~ estimate, data = dfm %>% dplyr::filter(child_expression > 500))))

    plot = ggplot(
        data = dfm, 
        aes(x = estimate, y = log2FoldChange, color = child_expression, label = label)) +
        ggrepel::geom_text_repel(max.overlaps = 15, parse = TRUE) +
        geom_point() +
        scale_colour_viridis_c() + 
        geom_smooth(method = "lm") +
        labs(x = "Direct effect estimate", y = "DESeq2 log2FCs", colour = "baseline expression\nof downstream gene") +
        cowplot::theme_cowplot(font_size = 12)

    fname = file.path(figure_dir(), "compare_log2fcs_direct_effect.pdf")
    ggsave(fname, plot, width = 8, height = 6, units = "in")

    fname = file.path(figure_dir(), "compare_log2fcs_direct_effect.png")
    ggsave(fname, plot, width = 8, height = 6, units = "in")
}



compare_dge = function(diffeq_with_pcs, diffeq_no_pcs, txdb) {
    
    dfm = dplyr::inner_join(
        diffeq_with_pcs %>%
            dplyr::select(readout_gene = gene, KO, log2FoldChange, padj),
        diffeq_no_pcs %>%
            dplyr::select(readout_gene = gene, KO, log2FoldChange, padj),
        by = c("readout_gene", "KO"),
        suffix = c("_with_pcs", "_no_pcs")
    ) %>%
        dplyr::filter(padj_with_pcs < 1e-10 | padj_no_pcs < 1e-10) %>%
        dplyr::inner_join(txdb, by = c("readout_gene" = "gene_id")) %>%
        dplyr::mutate(label = glue::glue("{KO}\U2192{gene_name}"))


    return(dfm)
}

plot_compare_dge = function(dfm) {

    fname = file.path(figure_dir(), "dge_results_compare_pca_effects.png")

    plot = ggplot(data = dfm, aes(x = log2FoldChange_with_pcs, y = log2FoldChange_no_pcs,
    label = label)) +
        ggrepel::geom_text_repel(max.overlaps = 30) +
        geom_point() +
        geom_smooth(method = "lm") +
        labs(x = "log2FCs after regressing out PCs 1-4", y = "log2FCs") +
        cowplot::theme_cowplot(font_size = 12)

    ggsave(fname, plot, width = 8, height = 6, units = "in")

    return(fname)
}

parse_results = function(dds_diffeq) {

    contrasts = resultsNames(dds_diffeq) %>%
        keep(~str_detect(.x, "KO"))

    pooled_result = map_dfr(
        contrasts,
        ~{
            results(dds_diffeq, name = .x) %>%
                as_tibble(rownames = "gene") %>%
                mutate(KO = str_split(.x, "_")[[c(1,2)]])
        }
    )

    return(pooled_result)
}


filter_low_counts = function(dds) {

    count_min_threshold = 10
    count_sample_threshold = 5
    ind = rowSums(counts(dds) >= count_min_threshold) >= count_sample_threshold
    log_info("Filtering {length(ind)} genes to {sum(ind)} based on low counts")
    dds = dds[ind, ]

    return(dds)
}

meta_tag = function() {

    # meta = "1.0_2022_10_06_llc_scaled"
    # meta = "nol1_nodagpenalty_llc_scaled"
    # meta = "nol1_nodagpenalty_2022_10_12_llc_scaled"
    meta = "JH"
    return(meta)
}

direct_effects_dir = function() {
    "output/diffeq"
}

direct_effects_location = function() {
    # "/home/users/jweinstk/.julia/dev/InferCausalGraph/output/csv/parsed_chain_0.01_2022_06_06_hyttinen_inspired_scaled.csv"
    # meta = "1.0_2022_09_30_llc_scaled"
    meta = meta_tag()
    # glue::glue("{Sys.getenv('SCRATCH')}/InferCausalGraph/output/csv/parsed_chain_{meta}.csv")
    file.path(direct_effects_dir(), "csv", glue::glue("parsed_chain_{meta}.csv"))
}

total_effects_location = function() {
    meta = meta_tag()
    # glue::glue("{Sys.getenv('SCRATCH')}/InferCausalGraph/output/csv/total_effects_{meta}.csv")
    file.path(direct_effects_dir(), "csv", glue::glue("total_effects_{meta}.csv"))
}

julia_targets_location = function() {
    meta = meta_tag()
    # glue::glue("{Sys.getenv('SCRATCH')}/InferCausalGraph/output/txt/targets_{meta}.txt")
    file.path(direct_effects_dir(), "txt", glue::glue("targets_{meta}.txt"))
}

read_direct_effects = function(file = direct_effects_location()) {
    vroom::vroom(file)
}

read_total_effects = function(file = total_effects_location()) {
    targets = read_julia_targets()
    vroom::vroom(file, col_names = targets) %>%
        dplyr::mutate(row = targets) %>%
        tidyr::pivot_longer(-row, names_to = "col", values_to = "total_effect")
}

read_julia_targets = function(file = julia_targets_location()) {
    readLines(file)
}


write_out_experiment = function(
    meta,
    diffeq,
    pca,
    mashr,
    joint_downstream_model,
    results_regressed_pcs,
    results_no_pcs,
    module_genes,
    txdb) {

    output_files = c(
        "intervention" = file.path(txt_dir(), "intervention_indicator.tsv"),
        "covariates" = file.path(txt_dir(), "covariates.tsv"),
        "mashr_significant_rows" = file.path(txt_dir(), "significant_rows.tsv"),
        "vst_normalized_counts" = file.path(txt_dir(), "vst_normalized_counts.tsv"),
        "vst_normalized_counts_no_kos" = file.path(txt_dir(), "vst_normalized_counts_no_kos.tsv"),
        "vst_normalized_counts_only_kos" = file.path(txt_dir(), "vst_normalized_counts_only_kos.tsv"),
        "vst_normalized_counts_transpose" = file.path(txt_dir(), "vst_normalized_counts_transpose.tsv"),
        "joint_downstream_model" = file.path(txt_dir(), "joint_downstream_model_differential_expression.tsv"),
        "normalized_counts" = file.path(txt_dir(), "normalized_counts.tsv"),
        "txdb" = file.path(txt_dir(), "txdb.tsv"),
        "deg_results_regressed_pcs" = file.path(txt_dir(), "differential_expression_results_regressed_pcs.tsv"),
        "deg_results_no_pcs" = file.path(txt_dir(), "differential_expression_results.tsv"),
        "module_genes" = file.path(txt_dir(), "module_genes_including_downstream.tsv"),
        "positive_module_genes" = file.path(txt_dir(), "positively_regulated_module_genes_including_downstream.tsv")
    )

    scores = pca$x %>%
        tibble::as_tibble(rownames = "sample") %>%
        dplyr::inner_join(meta, by = "sample")

    meta_names = names(meta)
    #done
    kod_genes = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::select(gene_name = KO) %>%
        dplyr::distinct(.) %>%
        dplyr::inner_join(txdb, by = "gene_name")
    # done
    intervention_indicator = meta %>%
        dplyr::mutate(value = 1L) %>%
        dplyr::select(-Donor) %>%
        tidyr::pivot_wider(names_from = "KO", values_from = "value", values_fill = 0L) %>%
        dplyr::select(-sample, -any_of(meta_names))

    readr::write_tsv(intervention_indicator, output_files["intervention"])
    # done
    # samples x genes
    normalized = DESeq2::vst(diffeq, blind = FALSE) %>%
        SummarizedExperiment::assay(.) %>%
        t %>%
        tibble::as_tibble(rownames = "Sample")

    readr::write_tsv(normalized, output_files["vst_normalized_counts"])
    #done
    # samples x (genes - KO'd genes)
    normalized_no_kos = normalized %>%
        dplyr::select(-any_of(kod_genes$gene_id))

    readr::write_tsv(normalized_no_kos, output_files["vst_normalized_counts_no_kos"])

    # samples x (only KO'd genes)
    normalized_only_kos = normalized %>%
        dplyr::select(Sample, all_of(kod_genes$gene_id)) %>%
        dplyr::inner_join(
                          meta %>%
                              dplyr::select(Sample = sample, gene_name = KO),
                by = "Sample"
        ) %>%
        dplyr::left_join(txdb, by = "gene_name")

    for(g in kod_genes$gene_id) {
        kod_rows = which(normalized_only_kos$gene_id == g)
        kod_rows = kod_rows[!is.na(kod_rows)]
        normalized_only_kos[kod_rows, g] = 0
    }

    normalized_only_kos = normalized_only_kos %>%
        dplyr::select(-gene_name, -gene_id)

    readr::write_tsv(normalized_only_kos, output_files["vst_normalized_counts_only_kos"])
    #done
    covariates = meta %>%
        dplyr::mutate(value = 1L, Donor = glue::glue("{Donor}_{experiment_version}")) %>%
        dplyr::select(sample, Donor, value) %>%
        tidyr::pivot_wider(names_from = "Donor", values_from = "value", values_fill = 0L) %>%
        dplyr::inner_join(scores, by = "sample") %>%
        dplyr::filter(sample %in% normalized$Sample) %>%
        dplyr::select(
            Sample = sample,
            starts_with("Donor_"),
            PC1:PC4
        )

    readr::write_tsv(covariates, output_files["covariates"])
    # done
    # genes x samples
    normalized = DESeq2::vst(diffeq, blind = FALSE) %>%
        SummarizedExperiment::assay(.) %>%
        tibble::as_tibble(rownames = "gene_id")

    readr::write_tsv(normalized, output_files["vst_normalized_counts_transpose"])

    # done
    normalized = DESeq2::counts(diffeq, normalized = TRUE) %>%
        tibble::as_tibble(rownames = "gene_id")

    readr::write_tsv(normalized, output_files["normalized_counts"])


    results_regressed_pcs %>%
        dplyr::rename(gene_id = gene) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::select(
            gene_id,
            gene_name,
            KO,
            everything()
        ) %>%
        readr::write_tsv(output_files["deg_results_regressed_pcs"])

    results_no_pcs %>%
        dplyr::rename(gene_id = gene) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::select(
            gene_id,
            gene_name,
            KO,
            everything()
        ) %>%
        readr::write_tsv(output_files["deg_results_no_pcs"])

    readr::write_tsv(txdb, output_files["txdb"])

    significant = tibble::tibble(
        rows = mashr::get_significant_results(mashr[["mashr"]], thresh = 0.001)
    )

    readr::write_tsv(significant, output_files["mashr_significant_rows"])
### 여기서부터터
    module_genes$all_genes %>%
        tibble::enframe(.) %>%
        tidyr::unnest(cols = value) %>%
        dplyr::rename(cluster = name, gene_name = value) %>%
        readr::write_tsv(output_files["module_genes"])

    module_genes$only_positively_regulated %>%
        tibble::enframe(.) %>%
        tidyr::unnest(cols = value) %>%
        dplyr::rename(cluster = name, gene_name = value) %>%
        readr::write_tsv(output_files["positive_module_genes"])

    pm = ashr::get_pm(joint_downstream_model) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = joint_downstream_model$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "posterior_mean", -to)

    psd = ashr::get_psd(joint_downstream_model) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = joint_downstream_model$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "posterior_sd", -to)

    lfsr = ashr::get_lfsr(joint_downstream_model) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = joint_downstream_model$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "lfsr", -to) %>%
        dplyr::inner_join(psd) %>%
        dplyr::inner_join(pm)

    downstream_edges = lfsr %>%
        dplyr::select(from, to, posterior_mean, posterior_sd, lfsr)

    readr::write_tsv(downstream_edges, output_files["joint_downstream_model"])

    return(output_files)
}

compute_covariance_reorder = function(direct_effects, txdb) {
    fnames = c(
        "vst_normalized_counts_only_kos" = file.path(txt_dir(), "vst_normalized_counts_only_kos.tsv")
    )

    ko_expression = vroom::vroom(fnames[1])

    symbols = tibble::tibble(
        gene_id = setdiff(names(ko_expression), "Sample")
    ) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::pull(gene_name)

    names(ko_expression) = c("Sample", symbols)

    direct_effects_wide = tidyr::pivot_wider(
        direct_effects,
        names_from = col,
        values_from = estimate,
        row
    )

    direct_effects_wide = dplyr::select(direct_effects_wide, row, all_of(symbols))
    direct_effects_wide = direct_effects_wide[match(symbols, direct_effects_wide$row), ]

    direct_matrix = as.matrix(direct_effects_wide %>% dplyr::select(-row))
    diag(direct_matrix) = 0.0

    I = diag(ncol(direct_matrix))

    a = solve(I - direct_matrix)

    total_covar = t(a) %*% a

    colnames(total_covar) = symbols

    write.table(
        total_covar, 
        file.path(txt_dir(), "total_covariance.csv"),
        sep = ",",
        row.names = FALSE
    )

    return(total_covar)
}

remove_ensembl_version = function(x) {
    stopifnot(is.character(x))
    if(stringr::str_detect(x[1], ".")) {
        return(stringr::str_extract(x, pattern = "[^.]+"))
    } else {
        return(x)
    }
}

remove_ensembl_version_txdb = function(txdb) {

    stopifnot(is.data.frame(txdb))
    
    txdb %>%
        dplyr::mutate(gene_id = remove_ensembl_version(gene_id))
}

convert_ensembl_to_symbol = function(x, txdb) {
    txdb %>%
        dplyr::filter(gene_id %in% .env[["x"]]) %>%
        dplyr::pull(gene_name)
}
