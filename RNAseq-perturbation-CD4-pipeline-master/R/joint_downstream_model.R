joint_downstream_path = function() {
    date = "2024_11_22"
    fnames = c(
        "posterior_mean" = file.path(
            txt_dir(), glue::glue("posterior_mean_downstream_summary_stats_{date}.csv")),
        "posterior_var" = file.path(
            txt_dir(), glue::glue("posterior_var_downstream_summary_stats_{date}.csv")),
        "posterior_lfsr" = file.path(
            txt_dir(), glue::glue("posterior_lfsr_downstream_summary_stats_{date}.csv"))
    )
}

read_joint_downstream_model = function(txdb) {
    result = list(
        "PosteriorMean" = vroom::vroom(joint_downstream_path()["posterior_mean"]),
        "PosteriorSD"  = vroom::vroom(joint_downstream_path()["posterior_var"]) %>%
            dplyr::mutate(
                dplyr::across(tidyselect:::where(is.double), sqrt)
            ),

        "lfsr"  = vroom::vroom(joint_downstream_path()["posterior_lfsr"])
    )

    result_nm = names(result)

    # gene_id --> gene_name
    downstream_symbols = tibble::tibble(
            gene_name = setdiff(names(result[[1]]), "row")
        ) %>%
        dplyr::inner_join(txdb, by = "gene_name") %>%
        dplyr::pull(gene_name)

    intervention_symbols = tibble::tibble(
            gene_name = result[[1]]$row
        ) %>%
        dplyr::inner_join(txdb, by = "gene_name") %>%
        dplyr::pull(gene_name)

    result = result %>%
        purrr::map(function(.x) {

            mat = .x %>%
                dplyr::select(tidyselect:::where(is.numeric)) %>%
                t

            colnames(mat) = intervention_symbols

            return(mat)
        }) %>%
        setNames(result_nm)

    return(list(
        "result" = result,
        "readout_gene" = downstream_symbols,
        "intervention"  = intervention_symbols
    ))
}

matrix_factorization_path = function() {
    date = "2023_01_23"
    fnames = c(
        "L" = file.path(
            txt_dir(), "../", glue::glue("posterior_matrix_factorization_L_{date}.csv")),
        "F" = file.path(
            txt_dir(), "../", glue::glue("posterior_matrix_factorization_F_{date}.csv"))
    )
}

read_matrix_factorization = function(txdb) {

    result = list(
        "L" = vroom::vroom(matrix_factorization_path()["L"]),
        "F"  = vroom::vroom(matrix_factorization_path()["F"])
    )

    result_nm = names(result)

    downstream_symbols = tibble::tibble(
            gene_id = setdiff(names(result[[2]]), "row")
        ) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::pull(gene_name)

    intervention_symbols = tibble::tibble(
            gene_id = result[[1]]$row
        ) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::pull(gene_name)


    return(list(
        "result" = result,
        "readout_gene" = downstream_symbols,
        "intervention"  = intervention_symbols
    ))
}

enrich_loadings = function(factorization, txdb, thresholds = c(0.20, 0.25, 0.30), db = "MSigDB_Hallmark_2020", pvalue_threshold = 0.05) {

    stopifnot("F" %in% names(factorization$result)) 

    Fhat = factorization$result$F
    K = nrow(Fhat)

    grid = tidyr::expand_grid(
        K = 1:K,
        threshold = thresholds
    )

    print(grid)

    enrichments = purrr::map2_dfr(
        grid$K, grid$threshold,
        function(.x, .y) {
            logger::log_info(glue::glue("Now working on K = {.x} and value = {.y}"))
            result = Fhat %>%
                dplyr::slice(.x) %>%
                tidyr::pivot_longer(names_to = "gene_id", values_to = "value", -row) %>%
                dplyr::filter(abs(value) > .env[[".y"]]) %>%
                dplyr::inner_join(txdb) %>%
                dplyr::pull(gene_name)

            n_genes = length(result)

            result = result %>%
                enrichR::enrichr(databases = db) %>%
                purrr::pluck(1) %>%
                tibble::as_tibble(.) %>%
                dplyr::filter(Adjusted.P.value < .env[["pvalue_threshold"]]) %>%
                dplyr::arrange(desc(Combined.Score)) %>%
                dplyr::mutate(
                    program = glue::glue("L{.x}"),
                    value_threshold = .env[[".y"]],
                    n_genes = .env[["n_genes"]]
                )

            return(result)
        }
    )
        
    return(enrichments) 
}

process_loading_enrichments = function(loadings) {
    loadings = loadings %>%
        dplyr::group_by(program) %>%
        dplyr::arrange(desc(Combined.Score)) %>%
        dplyr::slice(1) %>%
        dplyr::ungroup(.) 

    lookup = tibble::tibble(
        # Term = c("Cholesterol Homeostasis", "Allograft Rejection"),
        # New_Term = c("p53 Pathway", "Mitotic Spindle")
        program = c(
            "L1",
            "L3",
            "L5",
            "L6",
            "L7", 
            "L8",
            "L10"
        ),
        New_Term = c(
            "Inflammatory Response",
            "mTORC1 Signaling + E2F Targets",
            "Cell Cycle Regulation + DNA Repair",
            "Interferon Alpha Response + TNF-alpha Signaling",
            "IL-2/STAT5 Signaling",
            "Cell Proliferation (MYC + AKT/mTOR)",
            "IL-6/JAK/STAT3 Signaling"
        )

    )

    loadings = loadings %>%
        dplyr::left_join(lookup) %>%
        dplyr::mutate(
            program_label = ifelse(is.na(New_Term), Term, New_Term),
            program_index = stringr::str_remove_all(program, "L"),
            program_label = glue::glue("{program_index}: {program_label}")
        )

    return(loadings)

}

plot_loadings = function(matrix_factorization, program_labels, txdb, meta, row_cluster_k = 10, col_cluster_k = 3) {


    L = matrix_factorization$result$L %>%
        dplyr::rename(gene_id = row) %>%
        dplyr::inner_join(txdb, by = "gene_id") %>%
        dplyr::select(-gene_id)

    L_names = L %>%
        dplyr::select(starts_with("L")) %>%
        names

    for (nm in L_names) {
        names(L)[names(L) == nm] = program_labels$program_label[program_labels$program == nm]
    }

    L_numeric = L %>%
        dplyr::select(-gene_name) %>%
        as.matrix

    rownames(L_numeric) = L$gene_name

    row_cluster = hclust(dist(as.matrix(L_numeric), method = "euclidean"), method = "ward.D")
    row_cluster_order = row_cluster$order


    row_cluster_labels = cutree(row_cluster, k = row_cluster_k) %>%
        tibble::enframe(.) %>%
        dplyr::rename(row = name, row_group = value) %>%
        dplyr::mutate(
            row = forcats::fct_relevel(
                    row, 
                    levels = rownames(L_numeric)[row_cluster_order]
                )
        )

    col_cluster = hclust(dist(t(as.matrix(L_numeric)), method = "euclidean"), method = "ward.D")
    col_cluster_order = col_cluster$order

    col_cluster_labels = cutree(col_cluster, k = col_cluster_k) %>%
        tibble::enframe(.) %>%
        dplyr::rename(col = name, col_group = value) %>%
        dplyr::mutate(
            col = forcats::fct_relevel(
                    col, 
                    levels = col_cluster$labels[col_cluster_order]
                )
        )

    # 70th percentile loading value
    quantile_threshold = .75
    estimate_threshold = L_numeric %>%
        as.vector %>%
        quantile(prob = quantile_threshold)

    meta = meta %>%
        add_gene_group_colors

    row_labels = row_cluster_labels %>%
        dplyr::inner_join(
            dplyr::distinct(meta, row = KO, row_gene_group = gene_group, row_color = color),
            by = "row"
        ) %>%
        dplyr::mutate(
            row_label = glue::glue("<span style='color:{row_color}'>{row}</span>"),
        ) %>%
        dplyr::pull(row_label)
    
    pairs = tibble::as_tibble(L_numeric, rownames = "row") %>%
        dplyr::select(row, dplyr::all_of(col_cluster$labels[col_cluster$order])) %>%
        tidyr::pivot_longer(names_to = "col", values_to = "estimate", -row) %>%
        dplyr::inner_join(
            dplyr::distinct(meta, row = KO, row_gene_group = gene_group, row_color = color),
            by = "row"
        ) %>%
        dplyr::mutate(
            row_label = glue::glue("<span style='color:{row_color}'>{row}</span>"),
        ) %>%
        dplyr::mutate(
            row_label = factor(
                    row_label, 
                    levels = row_labels[row_cluster_order]
                )
        ) %>%
        dplyr::mutate(
            col = forcats::fct_relevel(
                    col, 
                    levels = col_cluster$labels[col_cluster$order]
                ),
            # estimate = dplyr::if_else(estimate < estimate_threshold, 0, estimate)
            estimate = estimate > estimate_threshold
        )

    plot = ggplot(data = pairs, aes(y = row_label, x = col, fill = factor(estimate))) +
                geom_tile() +
                # scale_fill_gradient(low = "gray", high = "red") +
                # scale_fill_viridis_c()  +
                scale_fill_manual(values = c("TRUE" = "firebrick3", "FALSE" = "gray")) +
                cowplot::theme_cowplot(font_size = 10) +
                labs(fill = glue::glue(">{quantile_threshold * 100L}th percentile Loading")) +
                theme(
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    # axis.text.x = element_text(angle=45, vjust=1, hjust=1),
                    axis.text.x = element_text(angle = 65, vjust = 1, hjust = 1),
                    axis.text.y = ggtext::element_markdown(),
                )
                
    ggsave(
        file.path(figure_dir(), glue::glue("matrix_factorization_loadings_heatmap.pdf")), 
        plot, width = 8.5, height = 12, units = "in"
    )

    return(pairs)
}
