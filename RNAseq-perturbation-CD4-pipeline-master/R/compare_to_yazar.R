compare_to_yazar = function(cis, trans, constraint, meta) {
    
    dfm = cis %>%
        dplyr::left_join(trans)

    genes = meta %>%
        dplyr::distinct(gene_name = KO, gene_group) %>%
        dplyr::mutate(
            has_cis_eQTL = gene_name %in% dfm$gene_name_cis,
            has_trans_eQTL_05 = gene_name %in% dfm$gene_name_trans[dfm$qvalue_trans < 0.05],
            has_trans_eQTL_10 = gene_name %in% dfm$gene_name_trans[dfm$qvalue_trans < 0.10],
            has_trans_eQTL_20 = gene_name %in% dfm$gene_name_trans[dfm$qvalue_trans < 0.20],
            has_trans_eQTL_30 = gene_name %in% dfm$gene_name_trans[dfm$qvalue_trans < 0.30],
            has_trans_eQTL_50 = gene_name %in% dfm$gene_name_trans[dfm$qvalue_trans < 0.50],
        ) %>%
        dplyr::inner_join(constraint, by = c("gene_name"))

    plot = ggplot(genes, aes(y = has_cis_eQTL, x = pLI, fill = factor(stat(quantile)))) +
        ggridges::stat_density_ridges(
                geom = "density_ridges_gradient", 
                calc_ecdf = FALSE,
                quantiles = 3, 
                quantile_lines = TRUE
            ) +
        scale_fill_viridis_d(name = "Tertiles") +
        cowplot::theme_cowplot(font_size = 12) +
        labs(y = "gene has a cis-eQTL in CD4+ T cells", x = "Constraint (pLI)")

    ggsave(file.path(figure_dir(), "ridgeline_ciseQTL.pdf"), plot, width = 6, height = 4)

    return(
        list(
            "eqtls" = dfm,
            "meta" = genes
        )
    )
}
