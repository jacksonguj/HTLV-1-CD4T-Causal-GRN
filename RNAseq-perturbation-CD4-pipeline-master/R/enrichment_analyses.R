constraint_metrics_location = function() {
    "/oak/stanford/groups/pritch/users/jweinstk/resources/constraint_metrics/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"    
}

tj_constraint_metrics_location = function() {
    "/oak/stanford/groups/pritch/users/jweinstk/resources/constraint_metrics/features.112922.tuned_081822.ipsita_demo_med_cov_200_low_map_false_lof_flag_true_mean_prop_005.125.3.8.1.2.1.4.eb_misannot.1e-8.tsv"
}

pics_ai_location = function() {
    "/home/users/jweinstk/network_inference/data/single_cell/human_cell_atlas/gene_annotations/AIgenes.txt"
}

iei_location = function () {
    # "/home/users/jweinstk/network_inference/data/single_cell/human_cell_atlas/gene_annotations/iei_genes_2021_09_17.txt"
    "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/experiments/resources/IUIS-IEI-list-for-web-site-December-2019-003.tsv"
}


pics_ai_genes = function() {
    readLines(pics_ai_location())
}

read_pathfindR_results = function(path = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/rnaseq_pipeline/scripts/output/diffeq/pathfindR_results/20230123_pathfindR_joint_all_KO_paths.csv") {
    vroom::vroom(path)
}

read_iei = function() {
    # readLines(iei_location())
    vroom::vroom(iei_location()) %>%
        dplyr::mutate(
            `Genetic defect` = stringr::str_remove(`Genetic defect`, "\xa0"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(GOF\\)"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(MLL2\\)"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(GCS1\\)"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(TNFRSF5\\)"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(TNFSF5\\)"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " GOF"),
            `Genetic defect` = stringr::str_remove(`Genetic defect`, " \\(RECQL3\\)"),
            `Genetic defect` = stringr::str_trim(`Genetic defect`),
            `Genetic defect` = stringr::str_squish(`Genetic defect`)
        ) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "Unknown")) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "\\.")) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "\\+")) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "del")) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "Del")) %>%
        dplyr::filter(!stringr::str_detect(`Genetic defect`, "deletion"))
}

read_iei = memoise::memoise(read_iei)

trans_eqtls_yazar_location = function() {
    # "/oak/stanford/groups/pritch/users/jweinstk/resources/onek1k_yazar/table_16_onek1k_yazar.csv"
    # a directory of parquets
    "/oak/stanford/groups/pritch/users/jweinstk/resources/onek1k_yazar/OneK1K_raw_trans_eQTL_results/parquet"
}

cis_eqtls_yazar_location = function() {
    "/oak/stanford/groups/pritch/users/jweinstk/resources/onek1k_yazar/table_ciseqtl_onek1k_yazar.csv"
}

trans_eqtls_eqtlgen_location = function() {
    "/oak/stanford/groups/pritch/users/jweinstk/resources/eqtlgen/2018-09-04-trans-eQTLsFDR0.05-CohortInfoRemoved-BonferroniAdded.txt.gz"
}

# read_yazar_trans_eqtls = function() {
#     df = vroom::vroom(trans_eqtls_yazar_location()) %>%
#         dplyr::filter(stringr::str_detect(`Cell type`, "CD4")) %>%
#         dplyr::distinct(
#             `trans_gene_name` = `trans eGene`,
#             `cis_gene_name` = `cis eGene`
#         )
#
#     df
# }

read_yazar_cis_eqtls = function(meta, path = cis_eqtls_yazar_location()) {

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        unique %>%
        as.character

    df = arrow::read_csv_arrow(path, as_data_frame = FALSE) %>%
        dplyr::filter(stringr::str_detect(`Cell type`, "CD4")) %>%
        dplyr::select(
            cell_type = `Cell type`,
            gene_name_cis = `Gene ID`,
            gene_id_cis   = `Gene Ensembl ID`,
            SNP,
            CHR = Chromosome,
            POS = Position,
            spearman_cis = `rho correlation coefficient`,
            qvalue_cis = qvalue
        ) %>%
        dplyr::mutate(
            CHR_POS = glue::glue("{CHR}:{POS}")
        ) %>%
        dplyr::filter(gene_name_cis %in% targets) %>%
        dplyr::collect(.)

    return(df)
}

read_yazar_trans_eqtls = function(cis, path = trans_eqtls_yazar_location()) {

    logger::log_info("now reading in trans eQTLs from Yazar et al.")

    ds = arrow::open_dataset(
        path,
        partitioning = arrow::DirectoryPartitioning$create(
            arrow::schema(c("cell_type" = arrow::string()))
        )
    ) %>% 
        dplyr::select(
            gene_name_trans = gene,
            CHR_POS = snp,
            spearman_trans = corr.coeff,
            qvalue_trans = qvalue,
            FDR_trans = localFDR,
            cell_type
        ) %>%
        dplyr::filter(CHR_POS %in% unique(cis$CHR_POS)) %>%
        dplyr::collect(.) %>%
        dplyr::mutate(
            cell_type = dplyr::case_when(
                cell_type == "CD4all" ~ "CD4 Naive",
                cell_type == "CD4effCM" ~ "CD4 Effector",
            )
        )

    logger::log_info("done reading in trans eQTLs from Yazar et al.")

    return(ds)
}

evaluate_yazar_trans_eqtl_density = function(path = trans_eqtls_yazar_location()) {

    logger::log_info("now reading in trans eQTLs from Yazar et al.")

    ds = arrow::open_dataset(
        path,
        partitioning = arrow::DirectoryPartitioning$create(
            arrow::schema(c("cell_type" = arrow::string()))
        )
    )

    density = purrr::map_dfr(c(1.00, 0.30, 0.20, 0.10, 0.05, 0.01), ~{

        logger::log_info(glue::glue("now reading in trans-eQTLs at qvalue < {.x}"))

        ds %>%
            dplyr::select(
                gene_name_trans = gene,
                CHR_POS = snp,
                spearman_trans = corr.coeff,
                qvalue_trans = qvalue,
                FDR_trans = localFDR,
                cell_type
            ) %>%
            # dplyr::filter(qvalue_trans <= .env[[".x"]]) %>%
            dplyr::filter(FDR_trans < .env[[".x"]]) %>%
            dplyr::summarize(
                n_genes = dplyr::n_distinct(gene_name_trans),
                n_SNPs = dplyr::n_distinct(CHR_POS),
                n_rows = dplyr::n(),
                FDR_threshold = .x
            ) %>%
            dplyr::collect(.) 
            # dplyr::mutate(
            #     cell_type = dplyr::case_when(
            #         cell_type == "CD4all" ~ "CD4 Naive",
            #         cell_type == "CD4effCM" ~ "CD4 Effector",
            #     )
            # )
    }, .progress = TRUE)

    logger::log_info("done reading in trans eQTLs from Yazar et al.")

    return(density)
}

read_eqtlgen_trans_eqtls = function() {
    df = vroom::vroom(trans_eqtls_eqtlgen_location()) %>%
        dplyr::distinct(
            gene_name = GeneSymbol
        )

    return(df)
}

read_schmidt_cytokine_hits = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/schmidt_cytokine_hits/schmidt_table_s2_cytokine_hits.csv") {
    vroom::vroom(path, col_select = c(
        Gene, 
        Screen_Version,
        CRISPRa_or_i,
        CD4_or_CD8,
        Cytokine,
        Hit_Type,
        zscore, # positive regulators have a positive zscore
        LFC,
        FDR
    )) %>%
    dplyr::filter(CD4_or_CD8 == "CD4") %>%
    dplyr::rename(
        gene_name = Gene
    ) 
}

read_dice = function(path ="/oak/stanford/groups/pritch/users/jweinstk/resources/DICE/DICE_2018_S2_cell_type_specificity.csv") {

    logger::log_info("now reading in DICE immune reference")

    df = vroom::vroom(path) %>%
            dplyr::rename(
                cell_type = `Cell type`,
                gene_id = `Ensembl ID`,
                gene_name = `Gene ID`
            ) %>%
            remove_ensembl_verison

    logger::log_info("done")

    return(df)
}

pull_cell_type_specific_genes = function(dice, cell_type) {

    valid_types = c(
        "Memory TREG cells",
        "Naïve CD4+ T cells",
        "Naïve TREG cells",
        "TFH cells",
        "TH1 cells",
        "TH1/17 cells",
        "TH17 cells",
        "TH2 cells"
    )

    stopifnot(is.data.frame(dice) && "cell_type" %in% names(dice))

    cell_type = match.arg(cell_type, valid_types)

    dice %>%
        dplyr::filter(cell_type == .env[["cell_type"]]) %>%
        dplyr::pull(gene_name)
}


remove_ensembl_verison = function(df) {
    df %>%
        dplyr::mutate(
            gene_id = stringr::str_extract(gene_id, "[^.]+")
        )
}


identify_blood_specific_genes = function() {

    tissue_exclude = c(
        "Cells - Cultured fibroblasts",
        "Cells - EBV-transformed lymphocytes"
    )

    vroom::vroom("/oak/stanford/groups/pritch/users/jweinstk/resources/GTEX/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz", delim = "\t", skip = 2) %>%
        tidyr::pivot_longer(
            cols = `Adipose - Subcutaneous`:`Vagina`, 
            names_to = "non_blood_tissue",
            values_to = "expression"
        ) %>%
        dplyr::filter(!(non_blood_tissue %in% tissue_exclude)) %>%
        dplyr::group_by(gene_id = Name) %>%
        dplyr::summarize(
            max_non_blood_expression = max(expression, na.rm = TRUE),
            `Whole Blood` = `Whole Blood`[1],
            blood_non_blood_ratio = `Whole Blood` / max_non_blood_expression
        ) %>%
        dplyr::ungroup(.) %>%
        dplyr::distinct(.) %>%
        remove_ensembl_verison %>%
        dplyr::mutate(
            blood_non_blood_ratio = ifelse(is.infinite(blood_non_blood_ratio), 100L, blood_non_blood_ratio)
        ) %>%
        dplyr::filter(
            blood_non_blood_ratio >= 1.50
        ) %>%
        dplyr::pull(gene_id)
}


enrichment = function(ko_specific_effects, dbs = "MSigDB_Hallmark_2020") {
    # dbs = "KEGG_2021_Human"
    enrichR::setEnrichrSite("enrichr")  
    genes = names(ko_specific_effects)

    results = purrr::imap(
                    ko_specific_effects, 
                    function(.x, .y) {
                        result = enrichR::enrichr(as.character(.x), dbs)[[1]] %>%
                            tibble::as_tibble(.) %>%
                            dplyr::mutate(
                                intervention = .y
                            )
                        return(result)
                    }
                ) %>%
                purrr::keep(~nrow(.x) > 0) %>%
                dplyr::bind_rows(.)

    return(results)
}

# enrichment = memoise::memoise(enrichment)


plot_enrichments = function(enrichment, tag = "diffeq") {

    plot = enrichment %>%
        dplyr::filter(Adjusted.P.value < .05 & Odds.Ratio > 5) %>%
        ggplot(data = ., 
            aes(y = forcats::fct_reorder2(Term, intervention, Odds.Ratio, .desc = TRUE),
                x = log10(Odds.Ratio), fill = intervention)
        ) +
            geom_col(position = position_dodge2(preserve = "single")) +
            # scale_fill_brewer(palette = "Set1") + 
            cowplot::theme_cowplot(font_size = 12) + 
            labs(x = "log10(OR)") +
            theme(
                axis.title.y = element_blank()
            )

    ggsave(file.path(figure_dir(), glue::glue("enrichments_{tag}.pdf")), plot, width = 7, height = 5, units = "in")
}



disease_enrichment_of_coregulation = function(downstream_indegree, diffeq, gene_lookup) {
    mean_expression = DESeq2::vst(diffeq) %>%
                            SummarizedExperiment::assay(.) %>%
                            matrixStats::rowMeans2(.)

    dispersions = DESeq2::dispersions(diffeq)

    constraint = vroom::vroom(constraint_metrics_location(), col_select = c(gene, pLI)) %>%
        dplyr::rename(gene_name = gene)

    expression = gene_lookup %>%
        dplyr::select(-gene_name) %>%
        dplyr::mutate(
            expression = mean_expression,
            dispersion = dispersions
        )

    features = downstream_indegree %>%
        dplyr::inner_join(expression, by = "gene_id") %>%
        dplyr::left_join(constraint, by = "gene_name") %>%
        dplyr::mutate(
            is_iei = purrr::map_lgl(
                gene_name,
                ~.x %in% iei_genes()
            ),
            is_pics_ai = purrr::map_lgl(
                gene_name,
                ~.x %in% pics_ai_genes()
            )
        )

    return(features)
}

plot_strata = function(downstream_features) {
    plot = downstream_features %>%
        tidyr::pivot_longer(names_to = "annotation", values_to = "value", is_iei:is_pics_ai) %>%
        dplyr::mutate(
            label = dplyr::case_when(
                        annotation  == "is_iei" ~ "IEI genes",
                        annotation  == "is_pics_ai" ~ "PICS AI GWAS genes"
                    )
        ) %>%
        ggplot(data = ., aes(y = value, x = n, fill = factor(stat(quantile)))) +
            # ggridges::geom_density_ridges() + 
            ggridges::stat_density_ridges(
                  geom = "density_ridges_gradient", calc_ecdf = TRUE,
                      quantiles = 4, quantile_lines = TRUE
             ) +
            scale_fill_viridis_d(name = "Quartiles") +
            labs(x = "Coregulation (number of incoming connections)") +
            facet_wrap(~label) +
            cowplot::theme_cowplot(font_size = 12) +
            cowplot::panel_border() + 
            theme(axis.title.y = element_blank())


    ggsave(file.path(figure_dir(), "downstream_coregulation_ridgeplot.pdf"), plot, width = 7, height = 5, units = "in")
}




iei_genes = function() {
    read_iei() %>%
        dplyr::pull(`Genetic defect`)
}

iei_genes = memoise::memoise(iei_genes)

iei_gene_list = function() {
    iei = read_iei() 

    labels = stringr::str_remove(iei$`Major category`, "^Table [1-9] ") %>%
        unique

    iei_list = split(iei$`Genetic defect`, iei$`Major category`)  %>%
        purrr::set_names(labels)

    return(iei_list) 
}

iei_gene_list = memoise::memoise(iei_gene_list)

impute_mask = function(df) {
    df %>%
        dplyr::mutate(
            dplyr::across(where(is.integer), ~dplyr::coalesce(.x, 0L))
        )
}

iei_gene_mask = function(iei_gene_list) {
    purrr::imap(iei_gene_list, ~{
        tibble::tibble(
            SYMBOL = .x
        ) %>%
        dplyr::mutate(
            {{.y}} := 1L,
        ) %>%
        dplyr::distinct(.)
    }) %>%
    purrr::compact(.) %>%
    purrr::reduce(dplyr::full_join) %>%
    impute_mask
}

map_go_term_to_symbols = function(term) {
                AnnotationDbi::select(
                    org.Hs.eg.db::org.Hs.eg.db, 
                    keytype="GOALL", 
                    keys=term, 
                    columns="SYMBOL"
                ) %>%
                tibble::as_tibble(.) %>%
                dplyr::filter(ONTOLOGYALL == "BP") %>%
                dplyr::select(SYMBOL) %>%
                dplyr::mutate(
                    {{term}} := 1L,
                ) %>%
                dplyr::distinct(.)
}

possibly_map_go_term_to_symbols = purrr::possibly(map_go_term_to_symbols, NULL)
possibly_map_go_term_to_symbols = memoise::memoise(possibly_map_go_term_to_symbols)

go_immune_gene_mask = function() {
    parent_term = "GO:0002376"
    immune_go = annotate::getGOChildren(parent_term)[[1]]$Children

    loadNamespace("org.Hs.eg.db")

    print(immune_go)

    symbols = purrr::map(immune_go, ~{
                print(glue::glue(".x = {.x}"))
                possibly_map_go_term_to_symbols(.x)
        }) %>%
            purrr::compact(.) %>%
            purrr::reduce(dplyr::full_join) %>%
            impute_mask

    return(symbols)
}

create_gene_mask = function(gene_lookup, meta, exclude_KOs = TRUE) {

    gene_mask = iei_gene_mask(iei_gene_list()) %>%
        dplyr::full_join(go_immune_gene_mask()) %>%
        impute_mask

    gene_mask = dplyr::left_join(
        gene_lookup,
        gene_mask,
        by = c("gene_name" = "SYMBOL")
    ) %>%
        impute_mask %>%
        dplyr::mutate(shared = 1L)

    KOs = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        unique

    if(exclude_KOs) {
        gene_mask = gene_mask %>%
            dplyr::filter(!(gene_name %in% KOs))
    }

    return(gene_mask)
}

write_gene_mask = function(gene_mask) {

    output = file.path(txt_dir(), "gene_mask.tsv") 
    readr::write_tsv(gene_mask, output)
    return(output)
}

extract_expression_by_gene = function(results, txdb) {

    expression = results %>%
        dplyr::distinct(gene_id = gene, expression = baseMean) %>%
        dplyr::inner_join(txdb, by = "gene_id")

    return(expression)
}

forest_plot_causal_centrality = function(
    centrality, expression, meta, constraint, iei_genes, gwas_genes, trans_egenes, 
    constraint_measure = 'pLI', tag = "pLI") {

    centrality = centrality %>%
        dplyr::rename(gene_name = name) %>%
        dplyr::inner_join(constraint, by = "gene_name") %>%
        dplyr::inner_join(expression, by = "gene_name") %>%
        dplyr::mutate(
            expression = log1p(expression),
            is_iei = gene_name %in% iei_genes,
            is_gwas = gene_name %in% gwas_genes,
            is_trans_egene = gene_name %in% trans_egenes,
            gene_group = relevel(factor(gene_group), ref = "Control")
        )

    print(as.data.frame(centrality))

    logger::log_info("now running models")

    models = list(
        "Out-degree" = MASS::glm.nb(
            as.formula(
                glue::glue(
                    "`out_degree` ~ {constraint_measure} + gene_group + expression"
                )
            ),
            data = centrality,
            maxit = 1e4
        ),

        "In-degree" = MASS::glm.nb(
            as.formula(
                glue::glue(
                    "`in_degree` ~ {constraint_measure} + gene_group + expression"
                )
            ),
            data = centrality,
            maxit = 1e4
        ),

        "Degree" = MASS::glm.nb(
            as.formula(
                glue::glue(
                    "`degree` ~ {constraint_measure} + gene_group + expression"
                )
            ),
            data = centrality,
            maxit = 1e4
        )
        
        # "Betweenness" = MASS::glm.nb(
        #     `betweenness` ~ pLI + gene_group + expression,
        #     data = centrality,
        #     maxit = 1e3
        # )
    )

    models_no_group = list(
        "Out-degree" = MASS::glm.nb(
            as.formula(glue::glue("`out_degree` ~ {constraint_measure}")),
            data = centrality %>% dplyr::mutate(pLI = pLI * 10),
            maxit = 1e4
        ),

        "In-degree" = MASS::glm.nb(
            as.formula(glue::glue("`in_degree` ~ {constraint_measure}")),
            data = centrality %>% dplyr::mutate(pLI = pLI * 10),
            maxit = 1e4
        ),

        "Degree" = MASS::glm.nb(
            as.formula(glue::glue("`degree` ~ {constraint_measure}")),
            data = centrality %>% dplyr::mutate(pLI = pLI * 10),
            maxit = 1e4
        )

        # "Betweenness" = MASS::glm.nb(
        #     `betweenness` ~ pLI,
        #     data = centrality,
        #     maxit = 1e3
        # )
    )

    logger::log_info("done running models")

    tidy_models = purrr::imap_dfr(models, ~{
        broom::tidy(.x, exponentiate = FALSE, conf.int = TRUE) %>%
            dplyr::mutate(label = .y)
    })

    tidy_models_no_group = purrr::imap_dfr(models_no_group, ~{
        broom::tidy(.x, exponentiate = TRUE, conf.int = TRUE) %>%
            dplyr::mutate(label = .y)
    })

    logger::log_info("now plotting")

    plot = ggplot(
        data = tidy_models %>%
            dplyr::filter(term != "(Intercept)") %>%
            dplyr::mutate(
                significant = p.value < .05,
                term = str_remove_all(term, "TRUE"),
                term = str_remove_all(term, "gene_group"),
                term_label = case_when(
                   term == "is_trans_egene" ~ "Is an eQTLgen trans-eGene", 
                   term == "pLI" ~ "Constraint (pLI)",
                   term == "tj_constraint" ~ "Constraint (Shet)",
                   term == "expression" ~ "Expression at baseline",
                   TRUE ~ term
                ),
                term_label = forcats::fct_reorder(term_label, estimate, .desc = TRUE)
            )
            , 
        aes(
            y = term_label,
            x = estimate,
            xmin = conf.low,
            xmax = conf.high,
            color = significant
        )
    ) +
        facet_wrap(~label) +
        # scale_x_continuous(labels = scales::percent, breaks = scales::breaks_pretty(n = 3)) +
        # scale_x_continuous(trans = "log10", breaks = c(0.1, 1, 10, 100)) +
        cowplot::theme_cowplot(font_size = 12) +
        # cowplot::background_grid() + 
        cowplot::panel_border() +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        geom_pointrange() +
        labs(x = "log(fold increase in connections) (95% CI)", color = "pvalue < 0.05") +
        theme(
            axis.title.y = element_blank()
        )

    fname = file.path(figure_dir(), glue::glue("causal_network_forest_{tag}.pdf"))
    
    ggsave(fname, plot, width = 5.5, height = 3, units = "in")

    plot = ggplot(
        data = tidy_models_no_group %>%
            dplyr::filter(term != "(Intercept)") %>%
            dplyr::mutate(
                significant = p.value < .05,
                term = str_remove_all(term, "TRUE"),
                term = str_remove_all(term, "gene_group"),
                term_label = case_when(
                   term == "is_trans_egene" ~ "Is an eQTLgen trans-eGene", 
                   term == "pLI" ~ "Constraint (pLI)",
                   term == "expression" ~ "Expression at baseline",
                   TRUE ~ term
                ),
                term_label = forcats::fct_reorder(term_label, estimate, .desc = TRUE)
            )
            , 
        aes(
            y = label,
            x = estimate - 1,
            xmin = conf.low - 1,
            xmax = conf.high - 1,
            color = significant
        )
    ) +
        # facet_wrap(~label) +
        scale_x_continuous(labels = scales::percent, breaks = scales::breaks_pretty(n = 3)) +
        # scale_x_continuous(trans = "log10", breaks = c(0.1, 1, 10, 100)) +
        cowplot::theme_cowplot(font_size = 12) +
        cowplot::background_grid() + 
        cowplot::panel_border() +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        geom_pointrange() +
        labs(x = "Percent increase in connections associated\nwith an increase in pLI by 0.10 (95% CI)", color = "pvalue < 0.05") +
        theme(
            axis.title.y = element_blank()
        )

    fname = file.path(figure_dir(), glue::glue("causal_network_forest_no_group_covariate_{tag}.pdf"))
    
    ggsave(fname, plot, width = 5.5, height = 3, units = "in")
        
    return(list(
        "full_model" = tidy_models,
        "constraint" = tidy_models_no_group
    ))
}


forest_plot_downstream_indegree = function(
    downstream_indegree_by_group,
    expression, 
    meta,
    constraint,
    iei_genes,
    gwas_genes,
    trans_egenes,
    blood_specific_genes,
    constraint_measure = "pLI",
    tag = "diffeq_pLI") {


    downstream_indegree_wide = downstream_indegree_by_group %>%
        tidyr::pivot_wider(names_from = gene_group, values_from = n, values_fill = 0) %>%
        dplyr::mutate(
            total = `IEI Target` + `Control` + `IL2RA Regulators`,
            non_control = total - Control
        ) %>%
        dplyr::inner_join(constraint, by = "gene_name") %>%
        dplyr::inner_join(expression, by = "gene_name") %>%
        dplyr::mutate(
            expression = log1p(expression),
            is_iei = gene_name %in% iei_genes,
            is_gwas = gene_name %in% gwas_genes,
            is_trans_egene = gene_name %in% trans_egenes,
            is_blood_specific = gene_name %in% blood_specific_genes
        )

    models = list(
        "IEI Targets" = MASS::glm.nb(
            as.formula(glue::glue(

            "`IEI Target` ~ {constraint_measure} + is_iei + is_gwas + Control + is_trans_egene + is_blood_specific + expression"
            )),
            # `IEI Target` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "IL2RA Regulators" = MASS::glm.nb(
            as.formula(glue::glue(

            "`IL2RA Regulators` ~ {constraint_measure} + is_iei + is_gwas + Control + is_trans_egene + is_blood_specific + expression"
            )),
            # `IL2RA Regulators` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "IEI + IL2RA" = MASS::glm.nb(
            as.formula(glue::glue(

            "`non_control` ~ {constraint_measure} + is_iei + is_gwas + Control + is_trans_egene + is_blood_specific + expression"
            )),
            # `non_control` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "Control" = MASS::glm.nb(
            as.formula(glue::glue(
                "`Control` ~ {constraint_measure} + is_iei + is_gwas + is_trans_egene + is_blood_specific + expression"
            )),
            # `Control` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        )
    )

    tidy_models = purrr::imap_dfr(models, ~{
        broom::tidy(.x, exponentiate = TRUE, conf.int = TRUE) %>%
            dplyr::mutate(label = .y)
    })

    plot = ggplot(
        data = tidy_models %>%
            dplyr::filter(term != "(Intercept)") %>%
            dplyr::mutate(
                significant = p.value < .05,
                term = str_remove_all(term, "TRUE"),
                term_label = case_when(
                   term == "is_iei" ~ "Is an IEI gene", 
                   term == "Control" ~ "Incoming connections from control TFs", 
                   term == "is_gwas" ~ "Is a GWAS gene", 
                   term == "is_trans_egene" ~ "Is an eQTLgen trans-eGene", 
                   term == "is_tcell_specific" ~ "Is a T-cell specific gene", 
                   term == "is_blood_specific" ~ "Is a whole blood specific gene", 
                   term == "pLI" ~ "Constraint (pLI)",
                   term == "tj_constraint" ~ "Constraint (Shet)",
                   term == "expression" ~ "Expression at baseline",
                   TRUE ~ term
                ),
                term_label = forcats::fct_reorder(term_label, estimate, .desc = TRUE)
            )
            , 
        aes(
            y = term_label,
            x = estimate - 1,
            xmin = conf.low - 1,
            xmax = conf.high - 1,
            color = significant
        )
    ) +
        facet_wrap(~label) +
        scale_x_continuous(labels = scales::percent, breaks = scales::breaks_pretty(n = 4)) +
        cowplot::theme_cowplot(font_size = 12) +
        # cowplot::background_grid() + 
        cowplot::panel_border() +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        geom_pointrange() +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        labs(x = "Percentage increase in incoming connections (95% CI)", color = "pvalue < 0.05") +
        theme(
            axis.title.y = element_blank()
        )

    fname = file.path(figure_dir(), glue::glue("downstream_indegree_forest_{tag}.pdf"))
    
    ggsave(fname, plot, width = 8, height = 5, units = "in")
        
    return(tidy_models)
}

forest_plot_downstream_indegree_cell_type_specific = function(
    downstream_indegree_by_group,
    expression, 
    meta,
    TH1_genes,
    TH2_genes,
    TH17_genes,
    naive_TREG_genes,
    naive_CD4_genes,
    memory_TREG_genes,
    tag = "diffeq") {


    downstream_indegree_wide = downstream_indegree_by_group %>%
        tidyr::pivot_wider(names_from = gene_group, values_from = n, values_fill = 0) %>%
        dplyr::mutate(
            total = `IEI Target` + `Control` + `IL2RA Regulators`,
            non_control = total - Control
        ) %>%
        dplyr::inner_join(expression, by = "gene_name") %>%
        dplyr::mutate(
            expression = log1p(expression),
            is_TH1_specific = gene_name %in% TH1_genes,
            is_TH2_specific = gene_name %in% TH2_genes,
            is_TH17_specific = gene_name %in% TH17_genes,
            is_naive_CD4_specific = gene_name %in% naive_CD4_genes,
            is_naive_TREG_specific = gene_name %in% naive_TREG_genes,
            is_memory_TREG_specific = gene_name %in% memory_TREG_genes
        )

    models = list(
        "IEI Targets" = MASS::glm.nb(
            `IEI Target` ~ is_TH1_specific + is_TH2_specific + is_TH17_specific + is_naive_TREG_specific + is_naive_CD4_specific + is_memory_TREG_specific + Control  + expression,
            # `IEI Target` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "IL2RA Regulators" = MASS::glm.nb(
            `IL2RA Regulators` ~ is_TH1_specific + is_TH2_specific + is_TH17_specific + is_naive_TREG_specific + is_naive_CD4_specific + is_memory_TREG_specific + Control  + expression,
            # `IL2RA Regulators` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "IEI + IL2RA" = MASS::glm.nb(
            `non_control` ~ is_TH1_specific + is_TH2_specific + is_TH17_specific + is_naive_TREG_specific + is_naive_CD4_specific + is_memory_TREG_specific + Control  + expression,
            # `non_control` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        ),

        "Control" = MASS::glm.nb(
            `Control` ~ is_TH1_specific + is_TH2_specific + is_TH17_specific + is_naive_TREG_specific + is_naive_CD4_specific + is_memory_TREG_specific + expression,
            # `Control` ~ is_iei + expression,
            data = downstream_indegree_wide,
            maxit = 1e3
        )
    )

    tidy_models = purrr::imap_dfr(models, ~{
        broom::tidy(.x, exponentiate = TRUE, conf.int = TRUE) %>%
            dplyr::mutate(label = .y)
    })

    plot = ggplot(
        data = tidy_models %>%
            dplyr::filter(term != "(Intercept)") %>%
            dplyr::mutate(
                significant = p.value < .05,
                term = str_remove_all(term, "TRUE"),
                term_label = case_when(
                   term == "Control" ~ "Incoming connections from control TFs", 
                   term == "is_TH1_specific" ~ "Th1 enriched", 
                   term == "is_TH2_specific" ~ "Th2 enriched", 
                   term == "is_TH17_specific" ~ "Th17 enriched", 
                   term == "is_naive_TREG_specific" ~ "Naive Treg enriched", 
                   term == "is_naive_CD4_specific" ~ "Naive CD4+ enriched", 
                   term == "is_memory_TREG_specific" ~ "Memory Treg enriched", 
                   term == "is_CD4_STIM_specific" ~ "Enriched for genes responsive to stimulation", 
                   term == "expression" ~ "Expression at baseline",
                   TRUE ~ term
                ),
                term_label = forcats::fct_reorder(term_label, estimate, .desc = TRUE)
            )
            , 
        aes(
            y = term_label,
            x = estimate - 1,
            xmin = conf.low - 1,
            xmax = conf.high - 1,
            color = significant
        )
    ) +
        facet_wrap(~label) +
        scale_x_continuous(labels = scales::percent, breaks = scales::breaks_pretty(n = 4)) +
        cowplot::theme_cowplot(font_size = 12) +
        # cowplot::background_grid() + 
        cowplot::panel_border() +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        geom_pointrange() +
        labs(x = "Percentage increase in incoming connections (95% CI)", color = "pvalue < 0.05") +
        theme(
            axis.title.y = element_blank()
        )

    fname = file.path(figure_dir(), glue::glue("downstream_indegree_forest_cell_type_specific_{tag}.pdf"))
    
    ggsave(fname, plot, width = 8, height = 5, units = "in")
        
    return(tidy_models)
}
