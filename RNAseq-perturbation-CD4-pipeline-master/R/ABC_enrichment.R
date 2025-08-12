read_ABC_GRN = function(meta, ABC_path = "/oak/stanford/groups/pritch/users/jweinstk/resources/ABC/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.liftover.bed.gz", cell_type = "CD4-positive_helper_T_cell-ENCODE") {
    KOs = meta %>%
        dplyr::filter(!is_control) %>%
        dplyr::pull(KO) %>%
        unique

    headers = readLines("/oak/stanford/groups/pritch/users/jweinstk/resources/ABC/ABC_headers.txt") 

    ABC_GRN = vroom::vroom(ABC_path, delim = "\t", col_names = headers) %>%
        dplyr::filter(TargetGene %in% KOs) %>%
        dplyr::filter(CellType %in% .env[["cell_type"]])

    return(ABC_GRN)
}

read_DAC = function(file = "/oak/stanford/groups/pritch/users/jweinstk/perturbation_data/experiments/Supplementary_Data_2_ATAC_Seq_results.parquet", pval_threshold = 5e-2) {

    logger::log_info("now reading in ATAC peaks")

    df = arrow::read_parquet(file, as_data_frame = FALSE) %>%
        dplyr::rename(KO = sample) %>%
        dplyr::filter(padj < .env[["pval_threshold"]]) %>%
        tibble::as_tibble(.) 

    logger::log_info("done reading in ATAC peaks")

    df %>%
        dplyr::mutate(
            KO = stringr::str_remove_all(KO, " KO")
        ) 
}

merge_DAC_ABC_GRN = function(ABC_GRN, DAC) {

    DAC = DAC %>%
        dplyr::select(
            seqnames = peak_chr,
            start = peak_start,
            end = peak_end,
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
    dfm = plyranges::join_overlap_intersect(DAC, ABC_GRN) %>%
        GenomicRanges::as.data.frame(.) %>%
        tibble::as_tibble(.) %>%
        dplyr::distinct(row, col)

    return(dfm)
}
