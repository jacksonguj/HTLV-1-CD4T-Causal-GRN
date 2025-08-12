plot_pathway = function(joint_downstream, txdb, upstream_gene, pathway = "R-HSA-451927") {

    library(SBGNview)
    library(org.Hs.eg.db)

    pm = joint_downstream$result$PosteriorMean %>%
        tibble::as_tibble(rownames = "gene_id") %>%
        dplyr::mutate(gene_name = joint_downstream$readout_gene) %>%
        dplyr::select(gene_id, gene_name, all_of(upstream_gene)) %>%
        dplyr::rename(beta = {{upstream_gene}})

    lfsr = joint_downstream$result$lfsr %>%
        tibble::as_tibble(rownames = "gene_id") %>%
        dplyr::select(gene_id, all_of(upstream_gene)) %>%
        dplyr::rename(lfsr = {{upstream_gene}})

    pm = pm %>%
        dplyr::inner_join(lfsr, by = "gene_id") %>%
        dplyr::mutate(
            beta = ifelse(lfsr < .05, beta, 0)
        )

# pm_updated <- pm %>%
#    dplyr::select(-gene_id) %>%  # Remove the existing gene_id column
#      dplyr::inner_join(txdb, by = "gene_name") %>%  # Join with txdb by gene_name
#      dplyr::select(gene_id, gene_name, beta, lfsr)
    
    vals = pm$beta
    entrez = mapIds(
        org.Hs.eg.db, 
        keys = stringr::str_extract(pm$gene_id, "[^.]+"), 
        column = "ENTREZID", 
        keytype = "ENSEMBL"
    )
    names(vals) = entrez

    print(vals)

    logger::log_info("now loading sbgn files")
    data("pathways.info", "sbgn.xmls")
    logger::log_info("done loading sbgn files")

    obj = SBGNview(
        gene.data = vals,
        gene.id.type = "entrez",
        # sbgn.gene.id.type = "entrez",
        # pathway.name = pathway,
        input.sbgn = pathway,
        output.file = glue::glue("pathway_plot_{pathway}_{upstream_gene}"),
        min.gene.value = -round(max(abs(vals)), 2),
        max.gene.value = round(max(abs(vals)), 2),
        mid.gene.value = 0,
        col.gene.low = 'blue',
        col.gene.high = 'red',
        col.gene.mid = 'white',
        output.format = c("png", "pdf")
    )

    print(obj)


}
