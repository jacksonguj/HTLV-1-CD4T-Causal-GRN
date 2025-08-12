proliferation_location = function() {
    "/oak/stanford/groups/pritch/users/jweinstk/resources/proliferation_legut/table_s4_legut_t_cell_proliferation.csv"
}

read_proliferation = function() {
    vroom::vroom(proliferation_location(), col_select = c("gene_symbol", "stat", "padj", "log2FoldChange")) %>%
        dplyr::mutate(
            proliferation = padj < 5e-2
        ) %>%
        dplyr::select(
            gene_name = gene_symbol,
            proliferation,
            proliferation_log2FC = log2FoldChange,
            zscore = stat
        )
}

plot_proliferation = function(meta, proliferation) {
    fc_counts = meta %>%
        dplyr::distinct(gene_name = KO, gene_FC_counts) %>%
        tidyr::drop_na(.)

    dfm = proliferation %>%
        dplyr::inner_join(fc_counts, by = "gene_name")

    plot = ggplot(
            data = dfm, 
            aes(x = proliferation_log2FC, y = gene_FC_counts, color = proliferation, label = gene_name)) +
        geom_point() +
        # geom_smooth() + 
        cowplot::theme_cowplot() +
        ggrepel::geom_label_repel() +
        labs(x = "Proliferative effect", y = "Normalized cell counts", color = "Proliferative\neffect pvalue < .05")

    ggsave(
        file.path(figure_dir(), "proliferation_scatter.pdf"), plot,
        width = 6, height = 4, units = "in"
    )
}

plot_proliferation_ko_specific = function(ko_specific_effects, proliferation, meta, tag = "diffeq") {

    counts = purrr::map_dbl(ko_specific_effects, ~{
            sig_proliferation = proliferation$gene_name[proliferation$proliferation]
            sum(.x %in% sig_proliferation) 
        }) %>%
        tibble::enframe(.) %>%
        dplyr::arrange(desc(value)) %>%
        dplyr::inner_join(meta %>% dplyr::distinct(KO, gene_group), by = c("name" = "KO"))

    print(summary(MASS::glm.nb(value ~ gene_group, data = counts)))

    top_counts = counts %>%
        dplyr::slice(1:20) 

    normalized_counts = purrr::map_dbl(ko_specific_effects, ~{
            sig_proliferation = proliferation$gene_name[proliferation$proliferation]
            sum(.x %in% sig_proliferation) / length(.x)
        }) %>%
        tibble::enframe(.) %>%
        dplyr::arrange(desc(value)) %>%
        dplyr::inner_join(meta %>% dplyr::distinct(KO, gene_group), by = c("name" = "KO"))

    top_normalized_counts = normalized_counts %>%
        dplyr::slice(1:20) 

    print(summary(MASS::glm.nb(value ~ gene_group, data = normalized_counts)))

    fnames = c(
        "top" = file.path(figure_dir(), glue::glue("top_ranked_effects_on_proliferation_{tag}.pdf")),
        "top_normalized" = file.path(figure_dir(), glue::glue("top_normalized_ranked_effects_on_proliferation_{tag}.pdf")),
        "distribution" = file.path(figure_dir(), glue::glue("ridgeline_effects_on_proliferation_{tag}.pdf")),
        "distribution_normalized" = file.path(figure_dir(), glue::glue("normalized_ridgeline_effects_on_proliferation_{tag}.pdf"))
    )

    plot = ggplot(data = top_counts, aes(x = forcats::fct_reorder(name, value, .desc = TRUE), y = value, fill = gene_group)) +
            geom_col() +
            labs(y = "Number of perturbation specific effects on proliferation", fill = "Gene group") +
            scale_y_continuous(expand = expand_scale(mult = c(0.0, 0.0))) +
            add_gene_group_fill_ggplot2() +
            cowplot::theme_cowplot(font_size = 12) +
            theme(
                legend.position = "top",
                axis.title.x = element_blank(),
                axis.text.x = element_text(angle=45, vjust=1, hjust=1)
            ) 
            # guides(fill = "none")

    ggsave(fnames["top"], plot, width = 8, height = 6, units = "in")

    plot = ggplot(data = top_normalized_counts, aes(x = forcats::fct_reorder(name, value, .desc = TRUE), y = value, fill = gene_group)) +
            geom_col() +
            labs(y = "Number of perturbation specific effects on proliferation", fill = "Gene group") +
            scale_y_continuous(expand = expand_scale(mult = c(0.0, 0.0))) +
            add_gene_group_fill_ggplot2() +
            cowplot::theme_cowplot(font_size = 12) +
            theme(
                legend.position = "top",
                axis.title.x = element_blank(),
                axis.text.x = element_text(angle=45, vjust=1, hjust=1)
            ) 
            # guides(fill = "none")

    ggsave(fnames["top_normalized"], plot, width = 8, height = 6, units = "in")


    # plot = ggplot(data = counts, aes(y = gene_group, x = value, fill = factor(stat(quantile)))) +
    plot = ggplot(data = counts, aes(y = gene_group, x = value)) +
                ggridges::stat_density_ridges(
                        geom = "density_ridges_gradient", calc_ecdf = TRUE
                        # quantiles = 4, quantile_lines = TRUE
                ) +
                # scale_fill_viridis_d(name = "Quartiles") +
                scale_x_continuous(trans = "log1p") + 
                cowplot::theme_cowplot(font_size = 12) +
                cowplot::panel_border() +
                theme(axis.title.y = element_blank(), axis.title.x = element_blank())


    ggsave(fnames["distribution"], plot, width = 8, height = 6, units = "in")

    # plot = ggplot(data = normalized_counts, aes(y = gene_group, x = value, fill = factor(stat(quantile)))) +
    plot = ggplot(data = normalized_counts, aes(y = gene_group, x = value)) +
                ggridges::stat_density_ridges(
                        geom = "density_ridges_gradient", calc_ecdf = TRUE
                        # quantiles = 4, quantile_lines = TRUE
                ) +
                # scale_fill_viridis_d(name = "Quartiles") +
                # scale_x_continuous(trans = "log1p") + 
                cowplot::theme_cowplot(font_size = 12) +
                cowplot::panel_border() +
                theme(axis.title.y = element_blank(), axis.title.x = element_blank())


    ggsave(fnames["distribution_normalized"], plot, width = 8, height = 6, units = "in")
}


plot_cytokine_proliferation_effect = function(proliferation, cytokine_hits, mashr, meta, lfsr_threshold = set_lfsr_threshold()) {

    pm = ashr::get_pm(mashr) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "beta", -to) 

    lfsr = ashr::get_lfsr(mashr) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "lfsr", -to) %>%
        dplyr::inner_join(pm) %>%
        dplyr::inner_join(
            meta %>% 
                add_gene_group_colors %>%
                dplyr::distinct(KO, gene_group, color), 
            by = c("from" = "KO")
        )

    print(lfsr)

    edges = lfsr %>%
        dplyr::filter(lfsr < lfsr_threshold) %>%
        dplyr::inner_join(
            proliferation %>%
                dplyr::select(
                    gene_name, 
                    proliferation_zscore = zscore
                ),
            by = c("to" = "gene_name")
        ) %>%
        dplyr::inner_join(
            cytokine_hits %>%
                dplyr::filter(Cytokine == "IFNG") %>%
                dplyr::select(
                    gene_name,
                    IFNg_zscore = zscore
                ),
            by = c("to" = "gene_name")
        )

    summarized_edges = edges %>%
        dplyr::group_by(from, gene_group, color) %>%
        dplyr::summarize(
            proliferation = sum(proliferation_zscore * sign(beta), na.rm = T),
            IFNg = sum(IFNg_zscore * sign(beta), na.rm = T),
        )

    plot = ggplot(summarized_edges, aes(x = proliferation, y = IFNg, color = gene_group, label = from)) +
        geom_point() +
        add_gene_group_color_ggplot2() +
        ggrepel::geom_label_repel() +
        cowplot::theme_cowplot(font_size = 11) +
        labs(
            x = "Aggregate downstream effect on proliferation",
            y = "Aggregate downstream effect on IFNg secretion"
        )

    ggsave(
        file.path(figure_dir(), "joint_downstream_proliferation_IFNg_scatter.pdf"),
        plot,
        width = 7,
        height = 5,
        units = "in"
    )

    return(edges)
}

read_effectorness_genes = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/effectorness_gradient/effector_genes.txt") {
    readLines(path)
}

compute_effector_score = function(mashr, effector_genes, cytokine_hits, cluster_membership, meta, lfsr_threshold = set_lfsr_threshold()) {

    pm = ashr::get_pm(mashr) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "beta", -to) 

    lfsr = ashr::get_lfsr(mashr) %>%
        tibble::as_tibble(.) %>%
        dplyr::mutate(to = mashr$readout_gene) %>%
        tidyr::pivot_longer(names_to = "from", values_to = "lfsr", -to) %>%
        dplyr::inner_join(pm) %>%
        dplyr::inner_join(
            meta %>%
                add_gene_group_colors %>%
                dplyr::distinct(KO, gene_group, color), 
            by = c("from" = "KO")
        ) %>%
        dplyr::inner_join(
            cluster_membership %>% dplyr::select(from_cluster = cluster, from_main_cluster = main_cluster, gene_name),
            by = c("from" = "gene_name")
        )

    edges = lfsr %>%
        dplyr::filter(lfsr < .env[["lfsr_threshold"]])

    effector_summary = edges %>%
        dplyr::mutate(
            is_effector = to %in% effector_genes
        ) %>%
        dplyr::group_by(from, from_cluster, from_main_cluster) %>%
        dplyr::summarize(
            effector_summary = sum(beta * is_effector),
            effector_signed_summary = sum(sign(beta) * is_effector),
            effector_upregulated_summary = sum(sign(beta) == 1 & is_effector),
            effector_downregulated_summary = sum(sign(beta) == -1 & is_effector)
        ) %>%
        dplyr::ungroup(.)

    effector_summary_by_cluster = edges %>%
        dplyr::mutate(
            is_effector = to %in% effector_genes
        ) %>%
        dplyr::group_by(from_cluster, from_main_cluster) %>%
        dplyr::summarize(
            effector_summary = sum(beta * is_effector),
            effector_signed_summary = sum(sign(beta) * is_effector),
            effector_upregulated_summary = sum(sign(beta) == 1 & is_effector),
            effector_downregulated_summary = sum(sign(beta) == -1 & is_effector)
        ) %>%
        dplyr::ungroup(.)

    plot = effector_summary %>% 
        dplyr::filter(abs(effector_signed_summary) >= 3) %>%
        dplyr::mutate(
            from = forcats::fct_reorder(from, effector_summary)
        ) %>%
        ggplot(data = ., aes(y = from, x = effector_summary, fill = from_cluster)) +
            geom_col() +
            cowplot::theme_cowplot(font_size = 12) +
            labs(x = "Aggregate regulation of genes that\npromote an effector phenotype", fill = "Gene module") +
            theme(
                axis.title.y = element_blank()
            ) +
            scale_fill_manual(
                values = c(
                    "1A" = "#8dd3c7",
                    "1B" = "#ffffb3",
                    "1C" = "#bebada",
                    "2A" = "#fb8072",
                    "2B" = "#80b1d3",
                    "2C" = "#fdb462",
                    "3A" = "#b3de69",
                    "3B" = "#fccde5",
                    "4A" = "#d9d9d9"
                )
            )

    plot_cluster = effector_summary_by_cluster %>% 
        dplyr::mutate(
            from_cluster = forcats::fct_reorder(from_cluster, effector_summary)
        ) %>%
        ggplot(data = ., aes(y = from_cluster, x = effector_summary, fill = from_cluster)) +
            geom_col() +
            cowplot::theme_cowplot(font_size = 12) +
            labs(x = "Aggregate regulation of genes that\npromote an effector phenotype") +
            theme(
                axis.title.y = element_blank()
            ) +
            scale_fill_manual(
                values = c(
                    "1A" = "#8dd3c7",
                    "1B" = "#ffffb3",
                    "1C" = "#bebada",
                    "2A" = "#fb8072",
                    "2B" = "#80b1d3",
                    "2C" = "#fdb462",
                    "3A" = "#b3de69",
                    "3B" = "#fccde5",
                    "4A" = "#d9d9d9"
                )
            ) +
            guides(fill = "none")
        
    ggsave(
        file.path(figure_dir(), "effectorness_gradient.pdf"),
        plot,
        width = 5,
        height = 6,
        units = "in"
    )

    ggsave(
        file.path(figure_dir(), "effectorness_gradient_by_cluster.pdf"),
        plot_cluster,
        width = 5,
        height = 5,
        units = "in"
    )

    # spearman = 0.244, pvalue = 0.0244
    TNF_plot = effector_summary %>%
        dplyr::inner_join(cytokine_hits, by = c("from" = "gene_name")) %>%
        dplyr::filter(CD4_or_CD8 == "CD4" & Cytokine == "TNF") %>%
        ggplot(data = ., aes(x = effector_summary, zscore, colour = -log10(FDR), label = from)) +
            geom_point() +
            cowplot::theme_cowplot(font_size = 12) + 
            scale_colour_viridis_c() + 
            ggrepel::geom_text_repel() +
            labs(
                x = "Aggregate regulation of genes that\npromote an effector phenotype",
                y = "TNF screen zscore",
                colour = "-log10(FDR)"
            )

    ggsave(
        file.path(figure_dir(), "TNF_effectorness.pdf"),
        TNF_plot,
        width = 6,
        height = 4,
        units = "in"
    )

    # spearman = 0.325, pvalue = 0.00235
    IFNG_plot = effector_summary %>%
        dplyr::inner_join(cytokine_hits, by = c("from" = "gene_name")) %>%
        dplyr::filter(CD4_or_CD8 == "CD4" & Cytokine == "IFNG") %>%
        ggplot(data = ., aes(x = effector_summary, zscore, colour = -log10(FDR), label = from)) +
            geom_point() +
            cowplot::theme_cowplot(font_size = 12) + 
            scale_colour_viridis_c() + 
            ggrepel::geom_text_repel() +
            labs(
                x = "Aggregate regulation of genes that\npromote an effector phenotype",
                y = "IFNg screen zscore",
                colour = "-log10(FDR)"
            )

    ggsave(
        file.path(figure_dir(), "IFNG_effectorness.pdf"),
        IFNG_plot,
        width = 6,
        height = 4,
        units = "in"
    )

    return(effector_summary)
}
