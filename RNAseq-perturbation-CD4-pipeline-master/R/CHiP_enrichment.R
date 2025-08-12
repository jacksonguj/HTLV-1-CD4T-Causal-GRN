read_ChIP = function(manifest_file = "/oak/stanford/groups/pritch/users/jweinstk/resources/ChIPAtlas/CHiP_manifest.tsv") {
    manifest = vroom::vroom(manifest_file) %>%
        dplyr::group_by(antigen) %>%
        dplyr::slice(1)

    purrr::map2_dfr(manifest$file, manifest$antigen, ~read_ChIP_(.x, .y)) %>%
        dplyr::mutate(
            KO = dplyr::if_else(KO == "STAT5", "STAT5B", KO) # the antigen corresponds to a STAT5B antibody based on https://chip-atlas.org/view?id=SRX212433
        )
}

read_ChIP_ = function(file, antigen) {

    logger::log_info(glue::glue("Reading {antigen} from {file}"))

    bed = vroom::vroom(file, col_select = 1:3, col_names = c("seqnames", "start", "end")) %>%
        dplyr::mutate(KO = antigen)

    return(bed)
}

merge_ABC_ChIP_GRN = function(ABC_GRN, ChIP) {

    ChIP_map = ChIP %>%
        dplyr::select(
            seqnames,
            start,
            end,
            row = KO
        ) %>%
        plyranges::as_granges(.)

    ABC_GRN = ABC_GRN %>%
        dplyr::select(
            seqnames = CHROM, 
            start = START, 
            end = END,
            col = TargetGene # child node 
        ) %>%
        plyranges::as_granges(.)

    # A gene is a parent of another gene if it changes the chromatin in at least one ABC enhancer
    dfm = plyranges::join_overlap_intersect(ChIP_map, ABC_GRN) %>%
        GenomicRanges::as.data.frame(.) %>%
        tibble::as_tibble(.) %>%
        dplyr::distinct(row, col)

    return(dfm)
}
