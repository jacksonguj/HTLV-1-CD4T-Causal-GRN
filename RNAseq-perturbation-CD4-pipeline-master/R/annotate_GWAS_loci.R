read_RA_GWAS_loci = function(path = "data/GWAS_CATALOG/GWAS_RA_loci_ishigaki.csv") {
    vroom::vroom(path, col_select = c(`Chr.`, `Position`, Novel, `Gene name`)) %>%
        dplyr::rename(
            CHROM = `Chr.`,
            POS = Position,
            gene_name = `Gene name`
        ) %>%
        dplyr::mutate(
            multiple_loci = stringr::str_detect(gene_name, ",")
        ) %>%
        dplyr::mutate(
            gene_name = dplyr::case_when(
                !multiple_loci ~ stringr::str_extract(gene_name, "[^(]+"),
                TRUE ~ gene_name
            )
        )
}

read_MS_GWAS_loci = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/GWAS_CATALOG/GCST009597/MS_genes.txt") {
    gene_name = readLines(path)

    parsed= purrr::map_chr(gene_name, ~{
        is_intergenic = stringr::str_detect(.x, "dist")
        if(is_intergenic) {
            distances = stringr::str_extract_all(.x, "(?<=\\().+?(?=\\))") %>%
                unlist %>%
                stringr::str_remove_all("dist=") %>%
                as.numeric
            nearest_gene = which.min(distances)
            genes = stringr::str_split(.x, ",")[[1]] %>%
                stringr::str_extract("[^(]+")
            nearest_gene_name = genes[nearest_gene]
            return(nearest_gene_name)
                
        } else {

            return(stringr::str_extract(.x, "[^(]+"))
        }
    })

    return(tibble::tibble(
            gene_name = parsed
    ))
}

filter_for_phase_3_inhibitor = function(path) {
    vroom::vroom(path) %>%
        dplyr::filter(phase >= 3) %>%
        dplyr::filter(actionType == "Inhibitor") %>%
        dplyr::distinct(symbol) %>%
        dplyr::pull(symbol)
}

read_RA_drugs = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/OpenTargets/EFO_0000685-known-drugs.tsv") {
    filter_for_phase_3_inhibitor(path)
}


read_MS_drugs = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/OpenTargets/MONDO_0005301-known-drugs.tsv") {
    filter_for_phase_3_inhibitor(path)
}

subset_causal_network_to_RA_gwas = function(causal_network, meta, threshold = set_causal_threshold()) {
    
    RA_GWAS_loci = read_RA_GWAS_loci() %>%
        dplyr::pull(gene_name) %>%
        unique

    targets = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        unique %>%
        as.character

    targets = intersect(targets, RA_GWAS_loci)

    causal = causal_network %>%
        dplyr::filter(row %in% targets | col %in% targets) %>%
        dplyr::filter(abs(estimate) > threshold)

    all_nodes = unique(c(causal$row, causal$col))

    graph = create_tidy_graph(
        causal,
        meta 
    ) %>%
        tidygraph::activate(nodes) %>%
        dplyr::mutate(
            is_RA_loci = name %in% RA_GWAS_loci,
            col = ifelse(is_RA_loci, "goldenrod3", "black")
        )

    plot = ggraph(graph, layout = "linear")  +
                geom_edge_arc(
                    aes(colour = signed_weight),
                        arrow = arrow(
                                angle = 12,
                                length = unit(0.10, "inches"),
                                # ends = "last",
                                type = "closed"
                        ),
                        alpha = .4,
                        start_cap = circle(2.6, 'mm'),
                        end_cap = circle(2.6, 'mm')
                )  +
                scale_edge_colour_gradient2(low = "blue", mid = "gray", high = "red")+
                geom_node_text(
                    aes(label = name, color = I(col)),
                    size = 1.9,
                    # colour = "black", 
                    # family = "serif",
                    check_overlap = TRUE,
                    repel = TRUE
                ) + 
                theme_graph(base_family = "Helvetica")

    ggsave(
        file.path(figure_dir(), "RA_causal_network.pdf"), 
        plot, 
        width = 11, 
        height = 4, 
        units = "in"
    )
    return(causal)
}

read_cluster_membership = function(path = "output/diffeq/txt/cluster_membership.csv") {
    vroom::vroom(path) %>%
        dplyr::rename(
            gene_name = Gene,
            cluster = Cluster,
            main_cluster = Main,
            sub_cluster = Sub
        )
}

annotate_RA_GWAS_loci = function(mashr, cluster_membership, meta, lfsr_threshold = set_lfsr_threshold()) {
    RA_GWAS_loci = read_RA_GWAS_loci()

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
        dplyr::filter(lfsr < lfsr_threshold)

    dfm = dplyr::inner_join(
        RA_GWAS_loci %>% dplyr::filter(!multiple_loci), 
        lfsr, 
        by = c("gene_name" = "to")
    )

    dfm_summary = dfm %>%
        dplyr::add_count(gene_name) %>%
        dplyr::mutate(
            gene_name = forcats::fct_reorder(gene_name, n, .desc = FALSE)
        ) %>%
        dplyr::inner_join(
            cluster_membership,
            by = c("from" = "gene_name")
        )

    print(dfm_summary)

    plot_group = ggplot(data = dfm_summary, aes(y = gene_name, fill = gene_group)) +
        add_gene_group_fill_ggplot2() +
        geom_bar(position = "stack") +
        cowplot::theme_cowplot(font_size = 12) +
        scale_x_continuous(expand = c(0, 0)) +
        labs(x = "Number of incoming connections", y = "", fill = "Gene group") +
        theme(
            axis.title.y = element_blank()
        )

    plot_cluster = ggplot(data = dfm_summary, aes(y = gene_name, fill = cluster)) +
        geom_bar(position = "stack") +
        cowplot::theme_cowplot(font_size = 12) +
        scale_x_continuous(expand = c(0, 0)) +
        labs(x = "Number of incoming connections", y = "", fill = "Cluster") +
        theme(
            axis.title.y = element_blank()
        )

    ggsave(
        file.path(figure_dir(), "RA_GWAS_barplot_by_gene_group.pdf"),
        plot_group,
        width = 6,
        height = 8,
        units = "in"
    )

    ggsave(
        file.path(figure_dir(), "RA_GWAS_barplot_by_cluster.pdf"),
        plot_cluster,
        width = 6,
        height = 8,
        units = "in"
    )

    return(dfm)
}

annotate_MS_GWAS_loci = function(mashr, cluster_membership, meta, lfsr_threshold = set_lfsr_threshold()) {
    MS_GWAS_loci = read_MS_GWAS_loci()

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
        dplyr::filter(lfsr < lfsr_threshold)

    # cluster_out_degree = lfsr %>%
    #     dplyr::inner_join(
    #         cluster_membership,
    #         by = c("from" = "gene_name")
    #     ) %>%
    #     dplyr::group_by(cluster) %>%
    #     dplyr::summarize(out_degree = dplyr::n())
        

    dfm = dplyr::right_join(
        MS_GWAS_loci %>%
            dplyr::mutate(is_MS_locus = TRUE),
        lfsr, 
        by = c("gene_name" = "to")
    ) %>%
        dplyr::mutate(is_MS_locus = dplyr::coalesce(is_MS_locus, FALSE))

    dfm_summary = dfm %>%
        dplyr::add_count(gene_name) %>%
        dplyr::mutate(
            gene_name = forcats::fct_reorder(gene_name, n, .desc = FALSE)
        ) %>%
        dplyr::inner_join(
            cluster_membership,
            by = c("from" = "gene_name")
        ) 

    print(dfm_summary)

    model = glm(is_MS_locus ~ cluster + 1, data = dfm_summary, family = binomial()) %>%
        broom::tidy(exponentiate = TRUE, conf.int = TRUE)

    forest = ggplot(
            data = model %>% 
                dplyr::filter(term != "(Intercept)") %>%
                dplyr::mutate(
                    term = stringr::str_remove(term, "cluster"),
                    term = forcats::fct_reorder(term, estimate)
                ), 
        aes(y = term, x = estimate, xmin = conf.low, xmax = conf.high, colour = -log10(p.value))) +
        geom_vline(xintercept = 1.0, linetype = "dashed", alpha = .5, colour = "gray") +
        geom_pointrange() +
        cowplot::theme_cowplot(font_size = 11) +
        labs(x = "Odds-ratio (95% CI) of outgoing connections\nto MS GWAS loci", colour = "-log10(pvalue)") +
        scale_colour_viridis_c() + 
        theme(axis.title.y = element_blank(), legend.position = "bottom")

    ggsave(
        file.path(figure_dir(), "cluster_MS_loci_enrichment.pdf"),
        forest,
        width = 4, 
        height = 4,
        units = "in"
    )

    print(dfm_summary)

    plot_group = ggplot(data = dplyr::filter(dfm_summary, is_MS_locus), aes(y = gene_name, fill = gene_group)) +
        add_gene_group_fill_ggplot2() +
        geom_bar(position = "stack") +
        cowplot::theme_cowplot(font_size = 9) +
        scale_x_continuous(expand = c(0, 0)) +
        labs(x = "Number of incoming connections", y = "", fill = "Gene group") +
        theme(
            axis.title.y = element_blank()
        )

    plot_cluster = ggplot(data = dplyr::filter(dfm_summary, is_MS_locus), aes(y = gene_name, fill = cluster)) +
        geom_bar(position = "stack") +
        cowplot::theme_cowplot(font_size = 9) +
        scale_x_continuous(expand = c(0, 0)) +
        labs(x = "Number of incoming connections", y = "", fill = "Cluster") +
        theme(
            axis.title.y = element_blank(),
            legend.position = "bottom"
        )

    ggsave(
        file.path(figure_dir(), "MS_GWAS_barplot_by_gene_group.pdf"),
        plot_group,
        width = 7,
        height = 11,
        units = "in"
    )

    ggsave(
        file.path(figure_dir(), "MS_GWAS_barplot_by_cluster.pdf"),
        plot_cluster,
        width = 5,
        height = 10,
        units = "in"
    )

    return(dfm_summary)
}

plot_drug_network = function(mashr, drug_targets, blood_specific_genes, constraint, meta, tag, lfsr_threshold = set_lfsr_threshold()) {

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
        dplyr::filter(lfsr < lfsr_threshold)

    unconstrained_genes = constraint %>%
        dplyr::filter(tj_constraint <= 0.05) %>%
        dplyr::pull(gene_name)

    edges = lfsr %>%
        dplyr::filter(to %in% .env[["drug_targets"]]) %>%
        # dplyr::filter(from %in% .env[["blood_specific_genes"]]) %>%
        dplyr::filter(from %in% .env[["unconstrained_genes"]]) %>%
        dplyr::rename(signed_weight = beta) %>%
        dplyr::mutate(
            # signed_weight = ifelse(
            #     gene_group != "Control",
            #     signed_weight,
            #     0
            # ),
            weight = abs(signed_weight)
        )

    targets = unique(c(edges$from, edges$to))

    status = tibble::tibble(name = targets) %>%
        dplyr::left_join(
            meta %>% 
            add_gene_group_colors %>%
            dplyr::distinct(name = KO, gene_group, color),
            by = "name"
        ) %>%
        dplyr::mutate(
            gene_group = dplyr::coalesce(gene_group, "downstream"),
            color = dplyr::coalesce(color, "black"),
        )

    graph = tidygraph::tbl_graph(
        nodes = tibble::tibble(name = targets) %>% 
            dplyr::inner_join(status, by = "name"),

        edges = edges %>% 
            dplyr::select(from, to, signed_weight, weight)  
    ) %>%
        tidygraph::activate(nodes) %>%
        tidygraph::mutate(
            degree = tidygraph::centrality_degree(mode = "all")
        )

    plot = ggraph(graph, layout = "linear", circular = TRUE)  +
                geom_edge_arc(
                    aes(colour = signed_weight, alpha = (1.0 - weight)),
                        arrow = arrow(
                                angle = 12,
                                length = unit(0.10, "inches"),
                                # ends = "last",
                                type = "closed",
                        ),
                        # alpha = .4,
                        strength = 0.3,
                        start_cap = circle(2.6, 'mm'),
                        end_cap = circle(2.6, 'mm')
                )  +
                scale_edge_colour_gradient2(low = "blue", mid = "gray", high = "red") +
                geom_node_text(
                    aes(
                        label = name, 
                        y = y * 1.04,
                        x = x * 1.04,
                        color = I(color),
                        filter = degree >= 1
                    ),
                    # aes(label = name),
                    size = 2.5
                    # angle = 45,
                    # colour = "black", 
                    # family = "serif",
                    # check_overlap = TRUE,
                    # repel = TRUE
                ) + 
                theme_graph(base_family = "Helvetica") +
                theme(
                    legend.position="none",
                    plot.margin=unit(c(0,0,0,0),"cm")
                )

    ggsave(
        file.path(figure_dir(), glue::glue("{tag}_drug_network.pdf")), 
        plot, 
        width = 6, 
        height = 4, 
        units = "in"
    )

    return(graph)
}

read_in_immune_GWAS = function(pvalue_threshold = 5e-6) {
    root_folder = "/oak/stanford/groups/pritch/users/jweinstk/resources/finngen"
    gwas_suffix = ".gz"
    finemap_suffix = ".SUSIE_extend.cred.summary.tsv"
    prefix = "finngen_R8_"

    paths = tibble::tibble(
        basename = c(
            "CHRONNAS",
            "G6_MS",
            "J10_ASTHMA_EXMORE",
            "K11_UC_STRICT2",
            "L12_DERMATITISECZEMA",
            "L12_LUPUS",
            "L12_PSORIASIS",
            "RHEUMA_NOS",
            "T1D_WIDE",
            "AUTOIMMUNE"
        ),
        phenotype = c(
            "Crohn's disease",
            "Multiple sclerosis",
            "Asthma",
            "Ulcerative colitis",
            "Dermititis + eczema",
            "Lupus",
            "Psoriasis",
            "Rheumatoid arthritis",
            "Type 1 diabetes",
            "Any autoimmune disease"
        )
    ) %>%
        dplyr::mutate(
            gwas_basename = glue::glue("{prefix}{basename}{gwas_suffix}"),
            finemap_basename = glue::glue("{prefix}{basename}{finemap_suffix}"),
            gwas_path = file.path(root_folder, "summary_stats", gwas_basename),
            finemap_path = file.path(root_folder, "finemapping", finemap_basename)
        )

    logger::log_info("now reading in immune GWAS")
    num_workers = 8L
    # future::plan(future::multisession, workers = num_workers)
    # gwas = furrr::future_pmap_dfr(
    gwas = purrr::pmap_dfr(

       list(x = paths$gwas_path, y = paths$finemap_path, z = paths$phenotype),
       function(x, y, z) {

            logger::log_info(glue::glue("now reading in GWAS of {z}"))

            parquet_fname = stringr::str_replace(x, ".gz", ".parquet")
            if(!file.exists(parquet_fname)) {
                gwas = vroom::vroom(x) %>%
                    dplyr::rename(chrom = `#chrom`) %>%
                    dplyr::mutate(
                        v = glue::glue("{chrom}:{pos}:{ref}:{alt}")
                    ) %>%
                    dplyr::select(
                        rsid = rsids,
                        v,
                        nearest_genes,
                        mlogp
                    ) %>%
                    dplyr::mutate(phenotype = z) %>%
                    dplyr::filter(mlogp > -log10(.env[["pvalue_threshold"]]))

                arrow::write_parquet(gwas, parquet_fname)
            } else {
                gwas = arrow::read_parquet(parquet_fname)
            }

            logger::log_info(glue::glue("Identified {nrow(gwas)} rows"))

            finemap = vroom::vroom(y) %>%
                dplyr::select(
                    v,
                    gene_most_severe,
                    PIP = prob
                )

            gwas = dplyr::left_join(gwas, finemap, by = "v")

            logger::log_info(glue::glue("Identified {nrow(finemap)} finemapped variants"))

                # dplyr::filter(mlogp > -log10(.env[["pvalue_threshold"]]))
            return(gwas)
       }
    )
    future::plan(future::sequential)
    logger::log_info("done reading in immune GWAS")

    return(gwas)
}

pan_immune_GWAS_enrichment = function(immune_loci, cluster_membership, mashr, txdb, meta, lfsr_threshold = set_lfsr_threshold()) {

    phenotypes = unique(immune_loci$phenotype)
    immune_loci_wide = immune_loci %>%
        dplyr::filter(mlogp > -log10(5e-8)) %>%
        # dplyr::filter(!is.na(PIP)) %>%
        dplyr::distinct(gene_name = nearest_genes, phenotype) %>%
        # dplyr::filter(!stringr::str_detect(gene_name, "HLA-")) %>%
        dplyr::mutate(value = TRUE) %>%
        tidyr::pivot_wider(names_from = phenotype, values_from = value, values_fill = FALSE)

    print(immune_loci_wide)

    loci_counts = immune_loci_wide %>%
        dplyr::summarize(across(where(is.logical), sum)) %>%
        tidyr::gather(key = phenotype, value = loci)
    print(loci_counts)

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
        dplyr::filter(lfsr < lfsr_threshold)

    dfm = lfsr %>%
        dplyr::left_join(immune_loci_wide, by = c("to" = "gene_name")) %>%
        dplyr::mutate(
            across(any_of(phenotypes), ~dplyr::coalesce(.x, FALSE))
        ) %>%
        dplyr::inner_join(
            cluster_membership,
            by = c("from" = "gene_name")
        )

    print(dfm)
    models = purrr::map_dfr(
        loci_counts %>%
            dplyr::filter(loci >= 20) %>%
            dplyr::pull(phenotype),

        ~{
            glm(as.formula(glue::glue("`{.x}` ~ cluster + 1")), data = dfm, family = binomial()) %>%
                broom::tidy(exponentiate = TRUE, conf.int = TRUE) %>%
                dplyr::mutate(phenotype = .x)
        }
    ) %>%
        dplyr::filter(term != "(Intercept)")

    plot = models %>%
        dplyr::mutate(
            estimate = ifelse(p.value < .05, estimate, 1.0),
            term = stringr::str_remove_all(term, "cluster")
        ) %>%
        dplyr::filter(phenotype != "Any autoimmune disease") %>%
        dplyr::inner_join(loci_counts, by = "phenotype") %>%
        dplyr::mutate(phenotype = glue::glue("{phenotype} ({loci})")) %>%
        ggplot(data = ., aes(y = phenotype, x = term, fill = estimate)) +
            scale_fill_gradient2(low = "blue", mid = "gray", high = "red", midpoint = 1.0) +
            geom_tile() + 
            cowplot::theme_cowplot(font_size = 12) +
            labs(fill = "Enrichment") +
            theme(
                axis.title = element_blank()
            )

    ggsave(
        file.path(figure_dir(), "AI_cluster_enrichment_heatmap.pdf"),
        plot,
        width = 6,
        height = 4,
        units = "in"
    )

    return(list(
        "models" = models,
        "dfm" = dfm,
        "immune_loci_wide" = immune_loci_wide
    ))
}
