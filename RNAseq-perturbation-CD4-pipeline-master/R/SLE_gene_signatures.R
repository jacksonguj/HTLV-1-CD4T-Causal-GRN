disease_activity_signatures_nakano = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/SLE_gene_signatures/SLE_VS_LOW_DISEASE_ACTIVITIY.csv") {
    
    vroom::vroom(path)
}

disease_state_signatures_nakano = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/SLE_gene_signatures/SLE_VS_HEALTHY_CONTROLS.csv") {

    vroom::vroom(path)
}

compute_enrichments_nakano = function(SLE_signatures, module_genes, background) {
    
    modules = names(module_genes)

    SLE_signatures %>%
        dplyr::filter(FDR < 0.05) %>%
        dplyr::filter(logFC > 0) %>%
        dplyr::group_by(`Cell type`) %>%
        tidyr::nest() %>%
        dplyr::mutate(
            model = purrr::map(data, function(signature) {
                        purrr::map_dfr(modules, function(x) {
                            df = tibble::tibble(
                                gene = background
                            ) %>%
                                dplyr::mutate(
                                    is_SLE_activity_enriched = gene %in% signature$Gene,
                                    is_in_module = gene %in% module_genes[[x]]
                                )

                            m = glm(is_SLE_activity_enriched ~ is_in_module, data = df, family = binomial())
                            broom::tidy(m, conf.int = TRUE, exponentiate = TRUE) %>%
                                dplyr::mutate(module = x) %>%
                                dplyr::filter(term == "is_in_moduleTRUE")
                    })
                })
        ) %>%
        tidyr::unnest(model) %>%
        dplyr::mutate(
            padj = p.adjust(p.value, "BH")
        )
}

plot_module_nakano = function(estimates, tag = "SLE_high_activity") {

    cell_types = c(
        "Tfh",
        "Th1",
        "Th2",
        "Th17"
        # "Mem CD4",
        # "Naive CD4",
        # "Fr. I nTreg",
        # "Fr. II eTreg",
        # "Fr. III T"
    )

    plot = estimates %>%
        dplyr::filter(`Cell type` %in% .env[["cell_types"]]) %>%
        dplyr::mutate(
            # sig = 0 < conf.low | conf.high < 0,
            # sig = p.value < 0.05,
            sig = padj < 0.05,
            label = `Cell type`
        ) %>%
        ggplot(data = ., aes(y = label, x = estimate, xmin = conf.low, xmax = conf.high, colour = sig)) +
        geom_pointrange() +
        facet_wrap(~module, nrow = 5, ncol = 2) + 
        cowplot::theme_cowplot(font_size = 12) +
        cowplot::panel_border() + 
        labs(x = glue::glue("Enrichment of module in {stringr::str_replace_all(tag, '_', ' ')} signature (95% CI)"), y = "", colour = "BH adjusted pvalue < 0.05") +
        geom_vline(xintercept = 1, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        theme(axis.title.y = element_blank(), legend.position = "bottom")

    ggsave(
        filename = file.path(figure_dir(), glue::glue("{tag}_module_forest_SLE_nakano_enrichment_2023_05_03.pdf")),
        plot,
        width = 6,
        height = 5,
        units = "in"
    )
}

plot_module_nakano_combined = function(disease_state, disease_activity, tag = "SLE_high_activity") {

    estimates = disease_state %>%
        dplyr::mutate(
            analysis = "Disease state"
        ) %>%
        dplyr::bind_rows(
            disease_activity %>%
                dplyr::mutate(
                    analysis = "Disease activity"
                )
        ) %>%
        dplyr::mutate(
            padj = p.adjust(p.value, "BH")
        )

    cell_types = c(
        "Tfh",
        "Th1",
        "Th2",
        "Th17"
        # "Mem CD4",
        # "Naive CD4",
        # "Fr. I nTreg",
        # "Fr. II eTreg",
        # "Fr. III T"
    )

    plot = estimates %>%
        dplyr::filter(`Cell type` %in% .env[["cell_types"]]) %>%
        dplyr::mutate(
            # sig = 0 < conf.low | conf.high < 0,
            # sig = p.value < 0.05,
            sig = padj < 0.05,
            sig_label = dplyr::case_when(
               sig & analysis == "Disease state" ~ "Disease state",
               sig & analysis == "Disease activity" ~ "Disease activity",
               TRUE ~ "No enrichment"
            ),
            label = `Cell type`
        ) %>%
        ggplot(data = ., aes(y = label, x = estimate, xmin = conf.low, xmax = conf.high, colour = sig_label)) +
        geom_pointrange(position=position_dodge(width=c(0.6))) +
        facet_wrap(~module, nrow = 5, ncol = 2) + 
        cowplot::theme_cowplot(font_size = 12) +
        cowplot::panel_border() + 
        labs(x = glue::glue("Enrichment of module in {stringr::str_replace_all(tag, '_', ' ')} signature (95% CI)"), y = "", colour = "BH adjusted pvalue < 0.05\n") +
        geom_vline(xintercept = 1, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c(
            "Disease state" = "#c23121",
            "Disease activity" = "#fcbe03",
            "No enrichment" = "grey"
        )) +
        guides(colour=guide_legend(nrow=2,byrow=TRUE)) +
        theme(
              axis.title.y = element_blank(),
              legend.position="bottom",
              legend.box="vertical",
              legend.margin=margin()
        )

    ggsave(
        filename = file.path(figure_dir(), glue::glue("state_activity_module_forest_SLE_nakano_enrichment_2023_05_03.pdf")),
        plot,
        width = 8,
        height = 8,
        units = "in"
    )

    plot = estimates %>%
        dplyr::filter(`Cell type` %in% .env[["cell_types"]]) %>%
        dplyr::mutate(
            # sig = 0 < conf.low | conf.high < 0,
            # sig = p.value < 0.05,
            sig = padj < 0.05,
            sig_label = dplyr::case_when(
               sig & analysis == "Disease state" ~ "Disease state",
               sig & analysis == "Disease activity" ~ "Disease activity",
               TRUE ~ "No enrichment"
            ),
            label = `Cell type`
        ) %>%
        dplyr::filter(module == "4A") %>%
        ggplot(data = ., aes(y = label, x = estimate, xmin = conf.low, xmax = conf.high, colour = sig_label)) +
        geom_pointrange(position=position_dodge(width=c(0.3))) +
        cowplot::theme_cowplot(font_size = 12) +
        labs(x = glue::glue("Module 4 enrichment in {stringr::str_replace_all(tag, '_', ' ')} signature (95% CI)"), y = "", colour = "BH adjusted pvalue < 0.05\n") +
        geom_vline(xintercept = 1, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c(
            "Disease state" = "#c23121",
            "Disease activity" = "#fcbe03",
            "No enrichment" = "grey"
        )) +
        guides(colour=guide_legend(nrow=2,byrow=TRUE)) +
        theme(
              axis.title.y = element_blank(),
              legend.position="bottom",
              legend.box="vertical",
              legend.margin=margin()
        )

    ggsave(
        filename = file.path(figure_dir(), glue::glue("module_4A_state_activity_module_forest_SLE_nakano_enrichment_2023_05_03.pdf")),
        plot,
        width = 5,
        height = 7,
        units = "in"
    )
}
