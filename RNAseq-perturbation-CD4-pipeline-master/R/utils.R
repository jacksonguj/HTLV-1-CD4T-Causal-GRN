read_in_pub_ko = function() {
    path = "~/network_inference/data/experiments/Supplementary_Table_5_RNA_Seq_results.gz"
    df = vroom::vroom(path) %>%
        dplyr::mutate(sample = stringr::str_replace(sample, " KO", "")) %>%
        dplyr::select(
            row = sample, 
            col = gene_name,
            logFC,
            pvalue = adj.P.Val
        )

    df
}

plot_network = function(parsed_chain, output_dir = figure_output_dir()) {

    graph = create_tidy_graph(parsed_chain) 

    plot = ggraph(graph, "stress", bbox = 10) + 
    # plot = ggraph(graph, "linear", circular = TRUE) + 
    # plot = ggraph(graph, "centrality", cen = graph.strength(graph)) + 
                geom_edge_arc2(
                    aes(colour = signed_weight),
                        arrow = arrow(
                                angle = 15,
                                length = unit(0.13, "inches"),
                                # ends = "last",
                                type = "closed"
                        ),
                        end_cap = circle(2.5, 'mm'),
                        start_cap = circle(2.5, 'mm')
                ) +
                geom_node_point() +
                geom_node_text(
                    aes(label = name),
                    size = 6.5,
                    colour = "black", 
                    # family = "serif",
                    check_overlap = TRUE,
                    repel = TRUE
                ) + 
                labs(colour = "weight") +
                scale_edge_colour_gradient2(low = "deepskyblue2", mid = "gray", high = "red") + 
                theme_graph(base_family = "Helvetica")

    ggsave(
        file.path(output_dir, glue("ggplot_network_direct_total_{meta}_{filter_meta}.pdf")),
        plot,
        width = 10,
        height = 7,
        units = "in"
    )
}

plot_variance_explained = function(direct_effect_matrix, indirect_effect_matrix, total_effect_matrix, covs, output_dir = figure_output_dir()) {
    direct_var_explained = diag(t(direct_effect_matrix) %*% covs %*% direct_effect_matrix)
    indirect_var_explained = diag(t(indirect_effect_matrix) %*% covs %*% indirect_effect_matrix)
    total_var_explained = diag(t(total_effect_matrix) %*% covs %*% total_effect_matrix)

    stopifnot(all(colnames(direct_effect_matrix) == colnames(indirect_effect_matrix)))
    stopifnot(all(colnames(indirect_effect_matrix) == colnames(total_effect_matrix)))

    h2 = tibble(
        direct = direct_var_explained / total_var_explained,
        indirect = indirect_var_explained / total_var_explained,
        label = colnames(direct_effect_matrix)
    )

    print(
        glue(
            "On average, {mean(h2$direct)} variance is explained by direct effects"
        )
    )

    print(
        glue(
            "On average, {mean(h2$indirect)} variance is explained by indirect effects"
        )
    )

    h2_plot = ggplot(data = h2, aes(x = direct, y = indirect, label = label)) +
        cowplot::theme_cowplot(font_size = 12) +
        geom_point() + 
        geom_text_repel(max.overlaps = 5, parse = TRUE) +
        # ggforce::facet_zoom(xlim = c(0, 0.5)) +
        labs(x = "Variance explained from direct effects", y = "Variance explained from indirect effects", color = "") 

    ggsave(
        file.path(output_dir, glue("ggplot_variance_explained_scatter_{meta}_{filter_meta}.pdf")),
        h2_plot,
        width = 8,
        height = 6,
        units = "in"
    )
}

plot_latent_confouding = function(direct_effect_matrix, covs, output_dir = figure_output_dir()) {
    nv = ncol(direct_effect_matrix)
    ident = diag(nv)
    sigma = (ident - direct_effect_matrix) %*% covs %*% t(ident - direct_effect_matrix)
    colnames(sigma) = colnames(direct_effect_matrix)
    
    plot = sigma %>%
        as_tibble %>%
        mutate(row = colnames(direct_effect_matrix)) %>%
        pivot_longer(names_to = "col", values_to = "estimate", -row) %>%
        ggplot(data = ., aes(x = row, y = col, fill = estimate)) +
            geom_tile() +
            cowplot::theme_cowplot(font_size = 12) +
            theme(
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.text.x = element_text(angle = 45, vjust = 0.95, hjust = 1)
            ) +
            scale_y_discrete(limits = rev) + 
            labs(fill = "residual covariance") +
            scale_fill_gradient2(low = "blue", mid = "white", high = "red")

    ggsave(
        file.path(output_dir, glue("ggplot_residual_confounding_covariance_{meta}_{filter_meta}.pdf")),
        plot,
        width = 8,
        height = 6,
        units = "in"
    )
}
