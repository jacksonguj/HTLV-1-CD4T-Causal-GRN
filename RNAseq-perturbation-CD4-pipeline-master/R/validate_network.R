validate_network = function(meta, hbase, causal, threshold = set_causal_threshold()) {

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 

    complete_graph = tidyr::expand_grid(row = targets, col = targets) %>%
        dplyr::inner_join(causal, by = c("row", "col")) %>%
        dplyr::mutate(
            has_causal_edge = abs(estimate) > threshold
        ) %>%
        dplyr::left_join(
            hbase %>%
                dplyr::select(row = gene_name_1, col = gene_name_2) %>%
                dplyr::mutate(in_hbase_forward = 1L),
            by = c("row", "col")
        ) %>%
        dplyr::left_join(
            hbase %>%
                dplyr::select(col = gene_name_1, row = gene_name_2) %>%
                dplyr::mutate(in_hbase_reverse = 1L),
            by = c("row", "col")
        ) %>%
        dplyr::mutate(
            in_hbase = (!is.na(in_hbase_forward)) | (!is.na(in_hbase_reverse))
        )


    return(complete_graph)

}

validate_DAC_sub_network = function(meta, DAC_ABC_GRN, causal, threshold = set_causal_threshold()) {

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 

    IL2RA_targets = meta %>%
        dplyr::filter(gene_group == "IL2RA Regulators") %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 

    # 24 * 83 possible edges
    complete_graph = tidyr::expand_grid(row = targets, col = targets) %>%
        dplyr::filter(row %in% .env[["IL2RA_targets"]]) %>%
        dplyr::inner_join(causal, by = c("row", "col")) %>%
        dplyr::mutate(
            has_causal_edge = abs(estimate) > threshold
        ) %>%
        dplyr::left_join(
            DAC_ABC_GRN %>%
                dplyr::mutate(in_DAC_ABC_GRN = 1L),
            by = c("row", "col")
        ) %>%
        dplyr::mutate(in_DAC_ABC_GRN = dplyr::coalesce(in_DAC_ABC_GRN, 0))

    return(complete_graph)
}


validate_sub_network = function(meta, external_network, causal, threshold = set_causal_threshold()) {

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        as.character %>%
        unique 


    # 24 * 83 possible edges
    complete_graph = tidyr::expand_grid(row = targets, col = targets) %>%
        dplyr::inner_join(causal, by = c("row", "col")) %>%
        dplyr::mutate(
            has_causal_edge = abs(estimate) > threshold
        ) %>%
        dplyr::left_join(
            external_network %>%
                dplyr::mutate(in_validation_GRN = 1L),
            by = c("row", "col")
        ) %>%
        dplyr::mutate(in_validation_GRN = dplyr::coalesce(in_validation_GRN, 0))

    return(complete_graph)
}


iterate_hbase_thresholding = function(meta, hbase, causal) {
    # upper = max(abs(causal$estimate))
    upper = 0.6
    thresholds = seq(0.04, upper, by = 0.1)

    models = tibble::tibble(
            threshold = thresholds
        ) %>%
        dplyr::mutate(
            model = purrr::map(threshold, ~{
                graph = validate_network(meta, hbase, causal, .x)
                model = glm(in_hbase ~ has_causal_edge, data = graph, family = binomial())
                model
            }),
            tidy = purrr::map(model, ~broom::tidy(.x, conf.int = TRUE, exponentiate = TRUE)),
            tidy = purrr::map(tidy, ~dplyr::filter(.x, term == "has_causal_edgeTRUE"))
        ) %>%
        tidyr::unnest(tidy)

    plot = ggplot(data = models, 
        aes(y = factor(threshold), x = estimate, xmin = conf.low, xmax = conf.high, colour = -log10(p.value))) +
            geom_pointrange() +
            cowplot::theme_cowplot(font_size = 12) +
            scale_colour_viridis_c() +
            labs(x = "Odds-ratio (95% CI)", y = "Causal network thresholding parameter", colour = "-log10(pvalue)")

    ggsave(file.path(figure_dir(), "hbase_enrichment.pdf"), plot, width = 6, height = 4)

    return(models)
}

iterate_ABC_GRN_thresholding = function(meta, DAC_ABC_GRN, causal) {
    # upper = max(abs(causal$estimate))
    upper = .04
    thresholds = seq(0.01, upper, by = 0.005)

    models = tibble::tibble(
            threshold = thresholds
        ) %>%
        dplyr::mutate(
            model = purrr::map(threshold, ~{
                graph = validate_DAC_sub_network(meta, DAC_ABC_GRN, causal, .x)
                model = glm(in_DAC_ABC_GRN ~ has_causal_edge, data = graph, family = binomial())
                model
            }),
            tidy = purrr::map(model, ~broom::tidy(.x, conf.int = TRUE, exponentiate = TRUE)),
            tidy = purrr::map(tidy, ~dplyr::filter(.x, term == "has_causal_edgeTRUE"))
        ) %>%
        tidyr::unnest(tidy)

    plot = ggplot(data = models, 
        aes(y = factor(threshold), x = estimate, xmin = conf.low, xmax = conf.high, colour = -log10(p.value))) +
            geom_pointrange() +
            cowplot::theme_cowplot(font_size = 12) +
            scale_colour_viridis_c() +
            labs(x = "Odds-ratio (95% CI)", y = "Causal network thresholding parameter", colour = "-log10(pvalue)")

    ggsave(file.path(figure_dir(), "DAC_ABC_GRN_enrichment.pdf"), plot, width = 6, height = 4)

    return(models)
}

iterate_ChIP_thresholding = function(meta, CHiP_ABC_GRN, causal) {
    # upper = max(abs(causal$estimate))
    upper = .04
    thresholds = seq(0.01, upper, by = 0.005)

    models = tibble::tibble(
            threshold = thresholds
        ) %>%
        dplyr::mutate(
            model = purrr::map(threshold, ~{
                graph = validate_sub_network(meta, CHiP_ABC_GRN, causal, .x)
                model = glm(in_validation_GRN ~ has_causal_edge, data = graph, family = binomial())
                model
            }),
            tidy = purrr::map(model, ~broom::tidy(.x, conf.int = TRUE, exponentiate = TRUE)),
            tidy = purrr::map(tidy, ~dplyr::filter(.x, term == "has_causal_edgeTRUE"))
        ) %>%
        tidyr::unnest(tidy)

    plot = ggplot(data = models, 
        aes(y = factor(threshold), x = estimate, xmin = conf.low, xmax = conf.high, colour = -log10(p.value))) +
            geom_pointrange() +
            cowplot::theme_cowplot(font_size = 12) +
            scale_colour_viridis_c() +
            labs(x = "Odds-ratio (95% CI)", y = "Causal network thresholding parameter", colour = "-log10(pvalue)")

    ggsave(file.path(figure_dir(), "ChIP_ABC_GRN_enrichment.pdf"), plot, width = 6, height = 4)

    return(models)
}
