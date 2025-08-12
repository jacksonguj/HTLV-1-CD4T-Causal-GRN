apply_mashr = function(results, txdb) {

    results = dplyr::inner_join(results, txdb, by = c("gene" = "gene_id")) %>%
        dplyr::rename(intervention = KO)
    filtered_results = dplyr::filter(results, !(gene_name %in% unique(results$intervention)))
    logger::log_info(glue::glue("Removed self-KOs from results"))
    n_interventions = length(unique(results$intervention))
    stopifnot((nrow(results) - nrow(filtered_results)) == n_interventions ^ 2)

    wide_beta = tidyr::pivot_wider(filtered_results, values_from = log2FoldChange, names_from = intervention, id_cols = gene) #id_cols 추가
    wide_se = tidyr::pivot_wider(filtered_results, values_from = lfcSE, names_from = intervention, id_cols = gene)

    logger::log_info("running mashr now")
    mash_data = mashr::mash_set_data(
        wide_beta %>% 
            dplyr::select(-gene) %>%
            as.matrix,
        wide_se %>% 
            dplyr::select(-gene) %>%
            as.matrix
    )

    U_c = mashr::cov_canonical(mash_data)

    Sys.setenv("OMP_NUM_THREADS" = "1")
    marginal_mash_result = mashr::mash_1by1(mash_data)
    strong = mashr::get_significant_results(marginal_mash_result, .05)

    logger::log_info("running mashr cov_pca now")

    U_pca = mashr::cov_pca(mash_data, 4, subset = strong)

    logger::log_info("running mashr cov_ed now")

    # doesn't work...openMP error
    # U_ed = mashr::cov_ed(mash_data, U_pca, subset = strong)

    logger::log_info("done running mashr cov_ed now")

    logger::log_info("now running mashr now")
    # mash_result = mashr::mash(mash_data, c(U_c, U_ed))
    mash_result = mashr::mash(mash_data, c(U_c, U_pca))
    logger::log_info("done running mashr now")
    
    return(list(
        "mashr" = mash_result,
        "covs"  = c(U_c, U_pca),
        "gene_id" = wide_beta$gene
    ))
}

is_object_mashr = function(mashr) {
    if(length(mashr) == 3 & "mashr" %in% names(mashr)) {
        return(TRUE)
    } else if(length(mashr) == 3 & !("mashr" %in% names(mashr))) {
        return(FALSE)
    } else {
       stop("unclear object attributes") 
    }
}

extract_pairwise_effects = function(mashr) {
    
    if(is_object_mashr(mashr)) {
        m = mashr[["mashr"]]
    } else {
        m = mashr
    }
    pairwise = mashr::get_pairwise_sharing(m, factor = 0.3, lfsr_thresh = set_lfsr_threshold())
    return(pairwise)
}

plot_pairwise_effects = function(pairwise, meta, tag = "diffeq") {

    cluster = hclust(dist(pairwise, method = "euclidean"), method = "ward.D2")
    cluster_order = cluster$order

    meta = meta %>%
        add_gene_group_colors

    row_labels = tibble(
            row = rownames(pairwise)
        ) %>%
        dplyr::inner_join(
            dplyr::distinct(meta, row = KO, row_gene_group = gene_group, row_color = color),
            by = "row"
        ) %>%
        dplyr::mutate(
            row_label = glue::glue("<span style='color:{row_color}'>{row}</span>"),
        ) %>%
        dplyr::pull(row_label)

    threshold = 0.15

    pairs = tibble::as_tibble(pairwise, rownames = "row") %>%
        tidyr::pivot_longer(names_to = "col", values_to = "estimate", -row) %>%
        dplyr::inner_join(
            dplyr::distinct(meta, row = KO, row_gene_group = gene_group, row_color = color),
            by = "row"
        ) %>%
        dplyr::inner_join(
            dplyr::distinct(meta, col = KO, col_gene_group = gene_group, col_color = color),
            by = "col"
        ) %>%
        mutate(
            row_label = glue::glue("<span style='color:{row_color}'>{row}</span>"),
            col_label = glue::glue("<span style='color:{col_color}'>{col}</span>")
        ) %>%
        dplyr::mutate(
            row_label = factor(
                    row_label, 
                    levels = row_labels[cluster_order]
                ),
            col_label = factor(
                    col_label, 
                    levels = row_labels[cluster_order]
                )
        ) %>%
        dplyr::mutate(
            estimate = ifelse(row_label == col_label, 1, estimate),
            estimate = if_else(abs(estimate) < threshold, 0, estimate)
        )

    plot = ggplot(data = pairs, aes(x = row_label, y = col_label, fill = estimate)) +
                geom_tile() +
                # scale_fill_gradient(low = "gray", high = "red") +
                scale_fill_viridis_c()  +
                cowplot::theme_cowplot(font_size = 11) +
                labs(fill = "shared effects") +
                theme(
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    # axis.text.x = element_text(angle=45, vjust=1, hjust=1),
                    axis.text.x = ggtext::element_markdown(angle = 45, vjust = 1, hjust = 1),
                    axis.text.y = ggtext::element_markdown(),
                )
                
    ggsave(
        file.path(figure_dir(), glue::glue("effect_sharing_{tag}.pdf")), 
        plot, width = 15, height = 11, units = "in"
    )
}

set_causal_threshold = function() {
    .025
}

set_lfsr_threshold = function() {
    5e-3
}

compute_degree = function(causal_network, results, threshold = set_causal_threshold()) {

    in_degree = causal_network %>%
        # dplyr::filter(PIP > .5) %>%
        dplyr::filter(abs(estimate) > threshold) %>%
        dplyr::group_by(col) %>%
        dplyr::summarize(in_degree = n()) %>%
        dplyr::rename(gene = col)

    out_degree = causal_network %>%
        # dplyr::filter(PIP > .5) %>%
        dplyr::filter(abs(estimate) > threshold) %>%
        dplyr::group_by(row) %>%
        dplyr::summarize(out_degree = n()) %>%
        dplyr::rename(gene = row)

    degree = full_join(in_degree, out_degree, by = "gene") %>%
        mutate(
                in_degree = coalesce(in_degree, 0),
                out_degree = coalesce(out_degree, 0),
                degree = in_degree + out_degree
              )

    return(degree)
}



stratify_degree_by_gene_group = function(degree, meta) {
    dfm = degree %>%
        dplyr::rename(total_degree = degree) %>%
        dplyr::inner_join(meta %>% dplyr::select(gene = KO, gene_group))

    dfm_long = degree %>%
        dplyr::rename(total_degree = degree) %>%
        tidyr::pivot_longer(names_to = "type", values_to = "value", -gene) %>%
        dplyr::inner_join(meta %>% dplyr::select(gene = KO, gene_group))

    print(summary(lm(out_degree ~ gene_group, data = dfm)))
    print(summary(lm(in_degree ~ gene_group, data = dfm)))
    print(summary(lm(total_degree ~ gene_group, data = dfm)))

    plot = ggplot(data = dfm_long, aes(y = gene_group, x = value, fill = factor(stat(quantile)))) +
                ggridges::stat_density_ridges(
                        geom = "density_ridges_gradient", calc_ecdf = TRUE,
                        quantiles = 4, quantile_lines = TRUE
                ) +
                scale_fill_viridis_d(name = "Quartiles") +
                facet_wrap(~type, nrow = 1, scales = "free_x") +
                scale_x_continuous(trans = "log1p") + 
                cowplot::theme_cowplot(font_size = 12) +
                cowplot::panel_border() +
                theme(axis.title.y = element_blank(), axis.title.x = element_blank())

    ggsave(file.path(figure_dir(), "degree_density_by_group.pdf"), plot, width = 8, height = 6, units = "in")
}

compare_pairwise_to_direct_effects = function(pairwise, causal_network, tag = "diffeq", threshold = set_causal_threshold()) {

    pairs = tibble::as_tibble(pairwise, rownames = "row") %>%
        tidyr::pivot_longer(names_to = "col", values_to = "sharing", -row)

    dfm = pairs %>%
        inner_join(causal_network, by = c("row", "col")) %>%
        mutate(
              label = glue::glue("{row}%->%{col}"),
              parsed_label = map_chr(label, ~as.character(as.expression(.x)))
        ) %>%
        filter(abs(estimate) > threshold) 

    # print(summary(lm(sharing ~ I(abs(estimate)), data = dfm)))
    print(summary(lm(sharing ~ estimate, data = dfm)))

    plot = ggplot(data = dfm, aes(x = estimate, y = sharing, label = parsed_label)) +
                geom_point() +
                geom_smooth(method = "lm") + 
                ggrepel::geom_text_repel(max.overlaps = 5, parse = TRUE) +
                cowplot::theme_cowplot(font_size = 12) +
                labs(x = "Causal effect magnitude", y = "Shared effects downstream", colour = "positive effect") 
                
    ggsave(file.path(figure_dir(), glue::glue("causal_scatter_effect_sharing_{tag}.pdf")), plot, width = 8, height = 6, units = "in")
}

compare_pairwise_to_distance = function(pairwise, causal_network, meta, tag = "diffeq", threshold = set_causal_threshold()) {
    
    graph = create_tidy_graph(
        causal_network %>% 
            dplyr::filter(abs(estimate) > threshold),
        meta
    )

    distances = igraph::distances(graph, weights = NA) %>%
        tibble::as_tibble(rownames = "row") %>%
        tidyr::pivot_longer(names_to = "col", values_to = "distance", -row)

    pairs = tibble::as_tibble(pairwise, rownames = "row") %>%
        tidyr::pivot_longer(names_to = "col", values_to = "sharing", -row) %>%
        dplyr::filter(row != col) 

    dfm = pairs %>%
        inner_join(distances, by = c("row", "col")) %>%
        mutate(
              label = glue::glue("{row}%->%{col}"),
              parsed_label = map_chr(label, ~as.character(as.expression(.x)))
        ) 

    print(summary(lm(sharing ~ distance, data = dfm)))

    plot = ggplot(data = dfm, aes(x = factor(distance), y = sharing, label = parsed_label)) +
                geom_boxplot() + 
                geom_jitter(alpha = .1) +
                # geom_smooth(method = "lm") + 
                # ggrepel::geom_text_repel(max.overlaps = 5, parse = TRUE) +
                cowplot::theme_cowplot(font_size = 12) +
                labs(x = "Distance between genes", y = "Shared effects downstream", colour = "positive effect") 
                
    ggsave(file.path(figure_dir(), glue::glue("causal_scatter_effect_sharing_distance_{tag}.pdf")), plot, width = 8, height = 6, units = "in")
}


compare_degree_estimates = function(results, causal_network) {

    direct_effect_degree = compute_degree(causal_network) 

    downstream_degree = results %>%
        dplyr::rename(intervention = KO) %>%
        dplyr::filter(padj < 5e-2) %>%
        dplyr::count(intervention) %>%
        dplyr::rename(gene = intervention, downstream_degree = n)

    dfm = dplyr::inner_join(direct_effect_degree, downstream_degree, by = "gene")

    plot = ggplot(data = dfm, aes(x = out_degree, y = downstream_degree, label = gene)) +
                geom_point() +
                geom_smooth(method = "lm") + 
                scale_y_continuous(trans = "log10") +
                ggrepel::geom_text_repel(max.overlaps = 5, parse = TRUE) +
                cowplot::theme_cowplot(font_size = 12) +
                labs(x = "Causal network out-degree", y = "log10(Downstream effect degree)") 
                
    ggsave(file.path(figure_dir(), "causal_scatter_degree.pdf"), plot, width = 8, height = 6, units = "in")
}

# copied from https://github.com/stephenslab/gtexresults/blob/master/code/normfuncs.R#L10
het_norm = function(effectsize) {
    t(apply(effectsize,1,function(x){
      x/x[which.max(abs(x))]
    }))
}

get_readout_gene_lookup = function(filtered_dds, txdb) {

    readout_genes = rownames(DESeq2::counts(filtered_dds))
    lookup = tibble::tibble(
        "gene_id" = readout_genes
    ) %>%
        dplyr::inner_join(txdb, by = "gene_id")

    stopifnot(nrow(lookup) == length(readout_genes))
    return(lookup)
}

extract_ko_specific_effects = function(mashr, txdb, lfsr_threshold = set_lfsr_threshold()) {

    if(is_object_mashr(mashr)) {
        m = mashr[["mashr"]]
        rownames = tibble::tibble(
            gene_id = mashr[["gene_id"]]
        ) %>%
            dplyr::inner_join(txdb) %>%
            dplyr::pull(gene_name)
    } else {
        m = mashr
        rownames = mashr$readout_gene
    }

    # rows where a gene is "significant" at lfsr < .05 for at least one KO
    significant = mashr::get_significant_results(m, thresh = lfsr_threshold) 
    pm = ashr::get_pm(m) # read-out genes X KO genes
    sig = pm[significant, ]
    rownames = rownames[significant]


    pm_norm = het_norm(sig) # fold change compared to largest effect
    specific_idx = rowSums(pm_norm > 0.5) == 1
    specific_pm_norm = pm_norm[specific_idx, ]
    rownames = rownames[specific_idx]

    stopifnot(length(rownames) == nrow(specific_pm_norm))

    cols = colnames(pm_norm)

    if(is_object_mashr(mashr)) {
        ko_results = purrr::set_names(cols) %>%
            purrr::imap(~{
                rownames[specific_pm_norm[, .y] > 0.5]
            })

    } else {
        ko_results = purrr::set_names(cols) %>%
            purrr::imap(~{
                mashr$readout_gene[m$result$lfsr[, .y] < lfsr_threshold]
            })

    }
    return(ko_results)
}

plot_top_downstream_indegree = function(mashr, expression, meta, lfsr_threshold = set_lfsr_threshold()) {

    if(is_object_mashr(mashr)) {
        m = mashr[["mashr"]]
        rownames = tibble::tibble(
            gene_id = mashr[["gene_id"]]
        ) %>%
            dplyr::inner_join(txdb) %>%
            dplyr::pull(gene_name)
    } else {
        m = mashr
        rownames = mashr$readout_gene
    }

    # IL2RA = meta %>%
    #     dplyr::filter(gene_group == "IL2RA Regulators") %>%
    #     dplyr::pull(KO) %>%
    #     as.character


    lfsr = ashr::get_lfsr(mashr) %>%  #lfsr_matrix <- ashr::get_lfsr(mashr[["mashr"]])
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "lfsr", -to)
    #lfsr = as.data.frame(lfsr_matrix) %>%
    #tibble::rownames_to_column(var = "to") %>%  # downstream gene 이름 추가
    #  tidyr::pivot_longer(cols = -to, names_to = "from", values_to = "lfsr")
    
    
    downstream_edges = lfsr %>%
        dplyr::filter(lfsr < lfsr_threshold) %>%
        # dplyr::filter(from %in% .env[["IL2RA"]]) %>%
        dplyr::select(from, to)

    counts = downstream_edges %>%
        dplyr::count(to) %>%
        dplyr::right_join(expression, by = c("to" = "gene_name")) %>%
        dplyr::mutate(n = dplyr::coalesce(n, 0)) %>%
        dplyr::arrange(desc(n))

    top_counts = counts %>%
        dplyr::slice(1:10)
    
    bar_plot = ggplot(data = top_counts, aes(x = forcats::fct_reorder(to, n, .desc = FALSE), y = n)) +
            geom_col(fill = "#c23121") +
            labs(y = "Number of incoming connections") +
            scale_y_continuous(expand = expand_scale(mult = c(0.0, 0.0))) +
            cowplot::theme_cowplot(font_size = 12) +
            add_gene_group_fill_ggplot2() +
            theme(
                legend.position = "top",
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.text.x = element_text(angle=45, vjust=1, hjust=1)
            ) 
            # guides(fill = "none")

    fname = file.path(figure_dir(), "top_10_incoming_connection_dist.pdf")
    ggsave(fname, bar_plot, width = 7, height = 5, units = "in")

    density_plot = ggplot(data = counts, aes(x = n)) +
            labs(fill = "", x = "Number of incoming connections", y = "Number of genes") +
            # ggridges::stat_density_ridges(
            #         geom = "density_ridges_gradient", 
            #         calc_ecdf = TRUE,
            #         quantiles = 1, quantile_lines = FALSE
            # ) +
            scale_y_continuous(expand = expand_scale(mult = c(0.0, 0.0)), labels = scales::comma) +
            scale_x_continuous(expand = expand_scale(mult = c(0.0, 0.0)), breaks = scales::pretty_breaks(n = 8)) +
            geom_histogram(fill = "gray37", aes(y = ..count..), binwidth = 1) +
            # ggridges::geom_density_ridges(
            #     jittered_points = TRUE,
            #     position = position_points_jitter(width = 0.10, height = 0.00),
            #     point_shape = '|',
            #     point_size = 4,
            #     point_alpha = 1
            # ) +
            cowplot::theme_cowplot(font_size = 10) +
            theme(
                legend.position = "top"
                # axis.title.y = element_blank()
                # axis.ticks.y = element_blank(),
                # axis.text.y = element_blank()
            ) +
            annotate(geom = "segment", x = 29, y = 0, xend = 29, yend = 100) +
            annotate(geom = "segment", x = 28, y = 0, xend = 28, yend = 100) +
            annotate(geom = "segment", x = 27, y = 0, xend = 27, yend = 100) +
            annotate(geom = "segment", x = 26, y = 0, xend = 26, yend = 100) 

    plot_with_inset = cowplot::ggdraw() +
        cowplot::draw_plot(density_plot)
        # cowplot::draw_plot(bar_plot, x= .4, y = .4, width = .5, height = .5)

    fname = file.path(figure_dir(), "incoming_connection_ridgeline.pdf")
    ggsave(fname, plot_with_inset, width = 5, height = 4, units = "in")

    return(counts)
}

parse_and_save_DEG_mashr = function(mashr, txdb, lfsr_threshold = set_lfsr_threshold()) {

    stopifnot(is_object_mashr(mashr))

    m = mashr[["mashr"]]

    # flip sign so positive effects = positive regulation
    m$result$PosteriorMean = m$result$PosteriorMean * -1

    rownames = tibble::tibble(
            gene_id = mashr[["gene_id"]]
        ) %>%
        dplyr::inner_join(txdb) %>%
        dplyr::pull(gene_name)

    pm = ashr::get_pm(m) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr[["gene_id"]]) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "pm", -to)

    psd = ashr::get_psd(m) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr[["gene_id"]]) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "psd", -to)

    lfsr = ashr::get_lfsr(m) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr[["gene_id"]]) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "lfsr", -to)

    downstream_edges = lfsr %>%
        dplyr::filter(lfsr < lfsr_threshold) %>%
        dplyr::inner_join(pm, by = c("from", "to")) %>%
        dplyr::inner_join(psd, by = c("from", "to")) %>%
        dplyr::left_join(txdb, by = c("to" = "gene_id")) %>%
        dplyr::rename(to_gene_symbol = gene_name) %>%
        dplyr::select(from, to, to_gene_symbol, posterior_mean = pm, posterior_sd = psd, lfsr)

    return(downstream_edges)
}

plot_ko_specific_effects = function(ko_specific_effects, meta, tag = NULL) {

    counts = purrr::map_dbl(ko_specific_effects, length) %>%
        tibble::enframe(.) %>%
        dplyr::arrange(desc(value)) %>%
        dplyr::inner_join(meta %>% dplyr::distinct(KO, gene_group), by = c("name" = "KO"))

    top_counts = counts %>%
        dplyr::slice(1:20) 

    print(counts)

    plot = ggplot(data = top_counts, aes(x = forcats::fct_reorder(name, value, .desc = FALSE), y = value, fill = gene_group)) +
            geom_col() +
            labs(y = "Number of perturbation specific effects", fill = "Gene group") +
            scale_y_continuous(expand = expand_scale(mult = c(0.0, 0.0))) +
            cowplot::theme_cowplot(font_size = 12) +
            add_gene_group_fill_ggplot2() +
            theme(
                legend.position = "top",
                axis.title.x = element_blank(),
                axis.text.x = element_text(angle=45, vjust=1, hjust=1)
            ) 
            # guides(fill = "none")

    if(is.null(tag)) {
        fname = file.path(figure_dir(), "ko_specific_effects.pdf")
    } else {
        fname = file.path(figure_dir(), glue::glue("ko_specific_effects_{tag}.pdf"))
    }

    ggsave(fname, plot, width = 9, height = 7, units = "in")

    print(counts, n= 30)

    plot = ggplot(data = counts, aes(y = gene_group, x = value, fill = gene_group)) +
            labs(fill = "", x = "Number of perturbation specific effects") +
            # ggridges::stat_density_ridges(
            #         geom = "density_ridges_gradient", 
            #         calc_ecdf = TRUE,
            #         quantiles = 1, quantile_lines = FALSE
            # ) +
            ggridges::geom_density_ridges() +
            cowplot::theme_cowplot(font_size = 12) +
            add_gene_group_fill_ggplot2() +
            theme(
                legend.position = "top",
                axis.title.y = element_blank()
            ) 
            # guides(fill = "none")

    if(is.null(tag)) {
        fname = file.path(figure_dir(), "ko_specific_effects_ridgeline.pdf")
    } else {
        fname = file.path(figure_dir(), glue::glue("ko_specific_effects_ridgeline_{tag}.pdf"))
    }

    ggsave(fname, plot, width = 6, height = 4, units = "in")

    plot = ggplot(data = counts, aes(y = gene_group, x = value, fill = gene_group)) +
            labs(fill = "", x = "Number of perturbation specific effects") +
            # ggridges::stat_density_ridges(
            #         geom = "density_ridges_gradient", 
            #         calc_ecdf = TRUE,
            #         quantiles = 1, quantile_lines = FALSE
            # ) +
            ggridges::geom_density_ridges(
                jittered_points = TRUE,
                position = position_points_jitter(width = 0.10, height = 0.00),
                point_shape = '|',
                point_size = 4,
                point_alpha = 1
            ) +
            cowplot::theme_cowplot(font_size = 12) +
            add_gene_group_fill_ggplot2(.8) +
            theme(
                legend.position = "top",
                axis.title.y = element_blank()
            )
            # guides(fill = "none")

    if(is.null(tag)) {
        fname = file.path(figure_dir(), "ko_specific_effects_ridgeline2.pdf")
    } else {
        fname = file.path(figure_dir(), glue::glue("ko_specific_effects_ridgeline2_{tag}.pdf"))
    }

    ggsave(fname, plot, width = 6, height = 4, units = "in")

    return(fname)
}

plot_covariance_pis = function(mashr) {

    pis = mashr::get_estimated_pi(mashr[["mashr"]]) %>%
            tibble::enframe(.) %>%
            dplyr::arrange(desc(value)) %>%
            dplyr::slice(1:10) %>%
            dplyr::mutate(
                label = dplyr::case_when(
                    name == "simple_het_1" ~ "Weakly shared effects",
                    name == "simple_het_2" ~ "Moderately shared effects",
                    name == "equal_effects" ~ "Identical effects",
                    name == "identity" ~ "Independent effects",
                    name == "tPCA" ~ "Truncated PCA",
                    name == "PCA_1" ~ "PC 1",
                    name == "PCA_2" ~ "PC 2",
                    name == "PCA_3" ~ "PC 3",
                    name == "PCA_3" ~ "PC 3",
                    name == "null" ~ "No effect",
                    TRUE ~ name
                ),
                label = forcats::fct_reorder(label, value, .desc = TRUE)
            )

    
    plot = ggplot(data = pis, aes(x = forcats::fct_reorder(label, value, .desc = TRUE), y = value, fill = label)) +
            geom_col() +
            labs(y = "Contribution of covariance matrices") +
            cowplot::theme_cowplot(font_size = 12) +
            theme(
                axis.title.x = element_blank(),
                axis.text.x = element_text(angle=45, vjust=1, hjust=1)
            ) + 
            guides(fill = "none")

    ggsave(file.path(figure_dir(), "estimated_pis.pdf"), plot, width = 8, height = 6, units = "in")
}

compute_downstream_indegree = function(mashr, txdb, lfsr_threshold = set_lfsr_threshold()) {

    lsfr = mashr[["mashr"]][["result"]][["lfsr"]] %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(gene_id = mashr[["gene_id"]]) %>%
        tidyr::pivot_longer(names_to = "intervention", values_to = "lfsr", -gene_id)

    lsfr %>%
        dplyr::filter(lfsr < lfsr_threshold) %>%
        dplyr::count(gene_id, sort = TRUE) %>%
        dplyr::inner_join(txdb, by = "gene_id")
}

# compute_downstream_indegree_by_group = function(mashr, gene_lookup, meta, lsfr_threshold = 5e-4) {
#
#     lsfr = mashr[["result"]][["lfsr"]] %>%
#         tibble::as_tibble(.) %>%
#         dplyr::mutate(gene_id = gene_lookup$gene_id) %>%
#         tidyr::pivot_longer(names_to = "intervention", values_to = "lsfr", -gene_id) %>%
#         dplyr::inner_join(
#             meta %>% dplyr::distinct(KO, gene_group), 
#             by = c("intervention" = "KO")
#         )
#
#     lsfr %>%
#         dplyr::filter(lsfr < lsfr_threshold) %>%
#         dplyr::count(gene_id, gene_group, sort = TRUE) %>%
#         dplyr::inner_join(gene_lookup, by = "gene_id")
# }

compute_downstream_indegree_by_group = function(mashr, txdb, meta, type, pm_threshold = 0.2, lfsr_threshold = set_lfsr_threshold()) {

    type = match.arg(type, c("pm", "lfsr"))

    if(is_object_mashr(mashr)) {
        m = mashr[["mashr"]]
        gene_ids = mashr[["gene_id"]]
    } else {
        m = mashr
        gene_ids = rownames(m$result$PosteriorMean)
    }

    if(type == "pm") {

        pm = ashr::get_pm(m) %>%
            tibble::as_tibble(.) %>%
            dplyr::mutate(gene_id = gene_ids) %>%
            tidyr::pivot_longer(names_to = "intervention", values_to = "pm", -gene_id) %>%
            dplyr::inner_join(
                meta %>% dplyr::distinct(KO, gene_group), 
                by = c("intervention" = "KO")
            )

        res = pm %>%
            dplyr::filter(abs(pm) > pm_threshold) %>%
            dplyr::count(gene_id, gene_group, sort = TRUE) %>%
            dplyr::inner_join(txdb, by = "gene_id")
    } else {

        lfsr = ashr::get_lfsr(m) %>%
            tibble::as_tibble(.) %>%
            dplyr::mutate(gene_id = gene_ids) %>%
            tidyr::pivot_longer(names_to = "intervention", values_to = "lfsr", -gene_id) %>%
            dplyr::inner_join(
                meta %>% dplyr::distinct(KO, gene_group), 
                by = c("intervention" = "KO")
            )

        res = lfsr %>%
            dplyr::filter(lfsr < lfsr_threshold) %>%
            dplyr::count(gene_id, gene_group, sort = TRUE) %>%
            dplyr::inner_join(txdb, by = "gene_id")
    }

    return(res)

}

predefined_enrichment = function(row_cluster_list, gene_sets) {
    stopifnot(!is.null(gene_sets))
    stopifnot(!is.null(row_cluster_list))

    all_gene_sets = unlist(gene_sets)
    all_cluster_genes = unlist(row_cluster_list)

    result = purrr::imap_dfr(row_cluster_list, function(cluster, cluster_label) {
        purrr::imap_dfr(gene_sets, function(set, set_label) {


            non_gene_set = setdiff(all_gene_sets, set)
            non_cluster_set = setdiff(all_cluster_genes, cluster)
            x11 = length(intersect(set, cluster))
            x12 = length(intersect(non_gene_set, cluster))
            x21 = length(intersect(set, non_cluster_set))
            x22 = length(intersect(non_gene_set, non_cluster_set))

            contingency_table = matrix(
                c(x11, x12, x21, x22),
                nrow = 2,
                byrow = TRUE
            )

            test = fisher.test(contingency_table)

            return(
                # column names chosen to mimic enrichr output
                tibble::tibble(
                    row_label = set_label,
                    row_group = which(cluster_label == names(row_cluster_list)),
                    Odds.Ratio = test$estimate,
                    Overlap = glue::glue("{x11}/{x12}"),
                    P.value = test$p.value
                )
            )
        })
    }) 

    print(result)
    stopifnot(is.data.frame(result) & nrow(result) > 0)

    result = result %>%
    dplyr::mutate(
        # row_group = match(row_index, names(row_cluster_list)),
        Adjusted.P.value = p.adjust(P.value, method = "BH")
    ) 

    print(result)
    return(result)
}

enrich_rows_predefined = function(row_cluster_labels, gene_lookup, min_overlap = 5, min_overlap_prop = .05, gene_sets) {

    row_cluster_nest = row_cluster_labels %>%
        # dplyr::inner_join(gene_lookup, by = c("row" = "gene_id")) %>%
        # dplyr::select(row_group, gene_name) %>%
        dplyr::select(row_group, gene_name = row) %>%
        dplyr::group_by(row_group) %>%
        tidyr::nest() %>%
        dplyr::ungroup(.)

    # named list of gene names. Names are the cluster identities
    row_cluster_list = row_cluster_nest$data %>%
                            purrr::set_names(row_cluster_nest$row_group) %>%
                            purrr::map(~dplyr::pull(.x, gene_name)) # convert single col dataframe to list
    row_enrichments = predefined_enrichment(row_cluster_list, gene_sets) 

    cluster_sizes = row_cluster_list %>%
        purrr::map_int(length) %>%
        tibble::enframe(.) %>%
        dplyr::rename(row_group = name, cluster_size = value)

    log_info("now printing cluster sizes")
    print(cluster_sizes)

    row_enrichments_single = row_enrichments %>%
        tidyr::separate(Overlap, into = c("overlap", "no_overlap"), sep = "/", remove = FALSE) %>%
        dplyr::mutate(across(c(overlap, no_overlap), as.numeric)) %>%
        dplyr::mutate(row_group = as.character(row_group)) %>%
        dplyr::inner_join(cluster_sizes, by = "row_group") %>%
        dplyr::group_by(row_group) %>%
        dplyr::mutate(
            group_min_overlap = pmin(cluster_size, min_overlap),
            group_overlap_prop = overlap / cluster_size
        ) %>%
        # dplyr::filter(overlap >= group_min_overlap & group_overlap_prop >= min_overlap_prop & Odds.Ratio > 1) %>%
        dplyr::arrange(Adjusted.P.value) %>%
        dplyr::slice(1) %>%
        dplyr::ungroup(.)

    stopifnot(nrow(row_enrichments_single) == length(unique(row_enrichments$row_group)))

    log_info("now printing cluster enrichments")
    print(row_enrichments_single %>% dplyr::select(row_label, Adjusted.P.value, row_group))

    return(row_enrichments_single)
}

enrich_rows_enrichr = function(row_cluster_labels, gene_lookup, min_overlap = 5, min_overlap_prop = 0.05, database = "GO_Biological_Process_2021") {
    row_cluster_nest = row_cluster_labels %>%
        # dplyr::inner_join(gene_lookup, by = c("row" = "gene_id")) %>%
        # dplyr::select(row_group, gene_name) %>%
        dplyr::select(row_group, gene_name = row) %>%
        dplyr::group_by(row_group) %>%
        tidyr::nest() %>%
        dplyr::ungroup(.)

    # browser()
    # named list of gene names. Names are the cluster identities

    row_cluster_list = row_cluster_nest$data %>%
                            purrr::set_names(row_cluster_nest$row_group) %>%
                            purrr::map(~dplyr::pull(.x, gene_name)) # convert single col dataframe to list

    row_enrichments = enrichment(row_cluster_list, database) %>%
        dplyr::rename(row_group = intervention)

    cluster_sizes = row_cluster_list %>%
        purrr::map_int(length) %>%
        tibble::enframe(.) %>%
        dplyr::rename(row_group = name, cluster_size = value)

    log_info("now printing cluster sizes")
    print(cluster_sizes)

    print(row_enrichments)
    
    row_enrichments_single = row_enrichments %>%
        tidyr::separate(Overlap, into = c("overlap", "no_overlap"), sep = "/", remove = FALSE) %>%
        dplyr::mutate(across(c(overlap, no_overlap), as.numeric)) %>%
        dplyr::inner_join(cluster_sizes, by = "row_group") %>%
        dplyr::group_by(row_group) %>%
        dplyr::mutate(
            group_min_overlap = pmin(cluster_size, min_overlap),
            group_overlap_prop = overlap / cluster_size
        ) %>%
        # dplyr::filter(overlap >= group_min_overlap & group_overlap_prop >= min_overlap_prop) %>%
        dplyr::arrange(Adjusted.P.value) %>%
        dplyr::slice(1) %>%
        dplyr::ungroup(.)

    log_info("now printing cluster enrichments")
    print(row_enrichments_single %>% dplyr::select(Term, Adjusted.P.value, row_group))

    stopifnot(nrow(row_enrichments_single) == length(unique(row_enrichments$row_group)))

    return(row_enrichments_single)
}

downstream_cluster = function(mashr, txdb, threshold = 5e-3, gene_names = NULL, gene_group_label = NULL, row_cluster_k = 8, col_cluster_k = 5, lower_limit_n_downstream = 1, tag = "diffeq", enrichment_function = "enrichr", ...) {

    if(is_object_mashr(mashr)) {
        rownames = tibble::tibble(
                gene_id = mashr[["gene_id"]]
            ) %>%
            dplyr::inner_join(txdb) %>%
            dplyr::pull(gene_name)

        m = mashr[["mashr"]]
        # flip interpretation such that positive coefficient means up-regulation
        m$result$PosteriorMean = m$result$PosteriorMean * -1
    } else {
        rownames = mashr$readout_gene
        m = mashr
    }

    # duplicated gene symbols
    duplicated_readouts = which(stringi::stri_duplicated(rownames))
    # rows where a gene is "significant" at lsfr < .05 for at least one KO
    significant_rows = mashr::get_significant_results(m, thresh = threshold) 
    idx = significant_rows
    idx = setdiff(idx, duplicated_readouts)
    mask = 1.0 * (ashr::get_lfsr(m) < threshold)
    pm = ashr::get_pm(m) # read-out genes X KO genes
    pm = pm * mask


    lower_limit_number_of_downstream_genes = 50

    enrichment_function = match.arg(enrichment_function, c("enrichr", "predefined"))

    if(!is.null(gene_names)) {
        sub_idx = which(rownames %in% gene_names)
        log_info(glue::glue("Identified {length(sub_idx)} in {gene_group_label} group"))
        idx = intersect(idx, sub_idx)
        log_info(glue::glue("Subsetting to {length(idx)} genes"))
    }

    sig = pm[idx, ]
    sum_by_intervention = colSums(mask[idx, ])
    sig = sig[, sum_by_intervention >= lower_limit_n_downstream]
    rownames = rownames[idx]
    rownames(sig) = rownames

    stopifnot(length(unique(rownames)) == length(rownames))

    row_cluster = hclust(dist(as.matrix(sig), method = "euclidean"), method = "ward.D")
    row_cluster_order = row_cluster$order

    row_cluster_labels = cutree(row_cluster, k = row_cluster_k) %>%
        tibble::enframe(.) %>%
        dplyr::rename(row = name, row_group = value) %>%
        dplyr::mutate(
            row = forcats::fct_relevel(
                    row, 
                    rownames(sig)[row_cluster_order] # levels = rownames(sig)[row_cluster_order]
                )
        )

    col_cluster = hclust(dist(t(as.matrix(sig)), method = "euclidean"), method = "ward.D")
    col_cluster_order = col_cluster$order

    col_cluster_labels = cutree(col_cluster, k = col_cluster_k) %>%
        tibble::enframe(.) %>%
        dplyr::rename(col = name, col_group = value) %>%
        dplyr::mutate(
            col = forcats::fct_relevel(
                    col, 
                    col_cluster$labels[col_cluster$order] # levels = col_cluster$labels[col_cluster$order]
                )
        )

    col_labels = col_cluster_labels %>%
        dplyr::group_by(col_group) %>%
        dplyr::summarize(col_label = paste(col, collapse = ", ")) %>%
        dplyr::mutate(
            col_label = stringr::str_wrap(col_label, width = 20)
        )

    col_cluster_labels = col_cluster_labels %>%
        dplyr::inner_join(col_labels, by = "col_group") %>%
        dplyr::arrange(col) %>% # order by levels of this factor
        dplyr::mutate(
            col_label = forcats::fct_inorder(col_label)
        )

    if(enrichment_function == "enrichr") {
        row_enrichment = enrich_rows_enrichr(row_cluster_labels, txdb) %>% # (row_cluster_labels, txdb, ...)
            dplyr::mutate(
                # removes extraneous go term prefix
                row_label = stringr::str_sub(
                    Term,
                    1L,
                    nchar(Term) - 13L
                )
            )
            
    } else {
        row_enrichment = enrich_rows_predefined(row_cluster_labels, txdb, ...)
    }
    log_info("here")
    print(row_enrichment)

    row_labels = row_enrichment %>%
        dplyr::mutate(
            row_label = stringr::str_wrap(row_label, width = 27),
            row_label = ifelse(
                Adjusted.P.value < 0.05,
                glue::glue("{row_label}*"),
                glue::glue("{row_label}")
            )
        ) %>%
        dplyr::mutate(
            row_group = as.integer(as.character(row_group))
        )

    row_cluster_labels = row_cluster_labels %>%
        dplyr::inner_join(row_labels, by = "row_group") %>%
        dplyr::arrange(row) %>% # order by levels of this factor
        dplyr::mutate(
            row_label = forcats::fct_inorder(row_label)
        )

    estimate_threshold = 0

    pairs = tibble::as_tibble(sig, rownames = "row") %>%
        dplyr::select(row, dplyr::all_of(col_cluster$labels[col_cluster$order])) %>%
        tidyr::pivot_longer(names_to = "col", values_to = "estimate", -row) %>%
        dplyr::mutate(
            row = forcats::fct_relevel(
                    row, 
                    # levels = rownames(sig)[row_cluster_order]
                    row_cluster$labels[row_cluster$order] # levels = row_cluster$labels[row_cluster$order]
                ),
            col = forcats::fct_relevel(
                    col, 
                    col_cluster$labels[col_cluster$order] # levels = col_cluster$labels[col_cluster$order]
                ),
            estimate = dplyr::if_else(abs(estimate) < estimate_threshold, 0, estimate)
        ) %>%
        dplyr::inner_join(row_cluster_labels, by = "row") %>%
        dplyr::inner_join(col_cluster_labels, by = "col")
        
    plot_downstream_cluster(pairs, gene_group_label, tag)
}

# library(patchwork)
plot_downstream_cluster = function(pairs, gene_group_label, tag) {

    heatmap = ggplot(data = pairs %>%
                dplyr::mutate(
                  estimate = case_when(
                    estimate == 0 ~ "no effect",
                    estimate < 0 ~ "down regulation",
                    estimate > 0 ~ "up regulation"
                  )
                ), 
                aes(x = col, y = row, fill = estimate)) +
                geom_tile() +
                # scale_fill_gradient2(low = "blue", mid = "gray", high = "red") +
                scale_fill_manual(
                    values = c(
                        "no effect" = "gray",
                        "up regulation" = "red",
                        "down regulation" = "#4E84C4"
                    )
                ) + 
                cowplot::theme_cowplot(font_size = 12) +
                labs(fill = "posterior mean effect") +
                theme(
                    axis.title = element_blank(),
                    axis.text  = element_blank(),
                    axis.ticks = element_blank(),
                    legend.position = "bottom"
                ) +
                guides(fill = guide_legend(
                                    title.position = "top", 
                                    title.hjust = 0.2, 
                                    keyheight = 0.3,
                                    keywidth = 0.4
                                )
                ) 


    col_plot = pairs %>%
        dplyr::distinct(col, col_group, col_label) %>%
        dplyr::mutate(dummy = 1) %>%
        ggplot(data = ., aes(y = dummy, x = col, fill = factor(col_label), label = col_label, group = col_label)) +
        geom_tile() + 
        theme_void() +
        labs(fill = "") + 
        theme(
            axis.title = element_blank(),
            axis.text = element_blank(),
            legend.position = "top"
        ) +  
        guides(fill = guide_legend(
                            title.position = "top", 
                            title.hjust = 0.5, 
                            keyheight = 0.4,
                            keywidth = 0.7
                        )
        ) 
        
    row_plot = pairs %>%
        dplyr::distinct(row, row_group, row_label) %>%
        dplyr::mutate(dummy = 1) %>%
        dplyr::mutate(row = forcats::fct_rev(row)) %>%
        ggplot(data = ., aes(x = dummy, y = row, fill = row_label, label = row_label)) +
        geom_tile() + 
        theme_void() +
        scale_fill_brewer(palette = "Set3") + 
        guides(color = guide_legend(
                            title.position = "top", 
                            title.hjust = 0.5, 
                            label.position = "left",
                            keyheight = 1.3,
                            keywidth = 0.4,
                            label.hjust = 0.5
                        )
        ) +
        labs(fill = "") + 
        theme(
            axis.title = element_blank(),
            axis.text = element_blank(),
            legend.position = "left",
            legend.text = element_text(size = 14)
        )  

    design = "
        #3
        12
    "
    plot = row_plot + heatmap + col_plot + patchwork::plot_layout(
        nrow = 2,
        ncol = 2,
        design = design,
        widths = c(0.1, 1.1),
        heights = c(0.05, 1.0)
    )
    if(is.null(gene_group_label)) {

        ggsave(file.path(figure_dir(), glue::glue("downstream_clustering_{tag}.pdf")), plot, width = 10.0, height = 8.0, units = "in")
    } else {

        fname = file.path(figure_dir(), glue::glue("downstream_clustering_{gene_group_label}_{tag}.pdf"))
        log_info(glue::glue("saving to {fname}"))
        ggsave(fname, plot, width = 10.0, height = 8.0, units = "in")
    }
}
