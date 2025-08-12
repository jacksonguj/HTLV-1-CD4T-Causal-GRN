suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(stringr)
    library(vroom)
    library(DESeq2)
    library(tximport)
    library(logger)
})


rna_seq_location = "/home/users/jweinstk/network_inference/data/experiments/RNA_UMI_dedup_counts.txt"
rna = vroom(rna_seq_location)

gencode_location = "/oak/stanford/groups/pritch/users/jweinstk/resources/gencode/gencode.v35.annotation.gtf.gz"

# txdb = GenomicFeatures::makeTxDbFromGFF(
#     gencode_location,
#     format = "gtf",
#     organism = "Homo sapiens",
#     dataSource = "gencode v35"
# )

txdb = rtracklayer::import(gencode_location) %>%
    as.data.frame %>%
    as_tibble %>%
    select(gene_id, gene_name) %>%
    distinct


sample_meta = tibble(
    sample = rna %>%
        dplyr::select(starts_with("Donor")) %>%
        names
) %>%
    mutate(
        donor = map_chr(
            sample,
            ~str_split(.x, "_")[[1]][[2]]
        ),
        intervention = map_chr(
            sample,
            ~str_split(.x, "_")[[1]][[3]]
        )
    )

count_matrix = rna %>% 
        dplyr::select(starts_with("Donor")) %>%
        as.matrix

rownames(count_matrix) = rna$Geneid

# dds = DESeqDataSetFromMatrix(
#     countData = count_matrix,
#     colData = sample_meta,
#     design = ~donor+intervention
#         
# )
# exclude low count genes
count_min_threshold = 10
count_sample_threshold = 5
ind = rowSums(counts(dds) >= count_min_threshold) >= count_sample_threshold
log_info("Filtering {length(ind)} genes to {sum(ind)} based on low counts")
# dds = dds[ind, ]

log_info("now running DESeq")
# dds_diffeq = DESeq(dds, parallel = TRUE, BPPARAM = BiocParallel::MulticoreParam(12))
# res = results(dds_diffeq)
contrasts = resultsNames(dds_diffeq) %>%
    keep(~str_detect(.x, "intervention"))

pooled_result = map_dfr(
    contrasts,
    ~{
        results(dds_diffeq, name = .x) %>%
            as_tibble(rownames = "gene") %>%
            mutate(intervention = str_split(.x, "_")[[c(1,2)]])
    }
)
log_info("done running DESeq")
# txi = tximport(
#     rna_seq_location,
#     type = "none",
#     tx2gene = txdb,
#     geneIdCol = "Geneid",
#     txIn = FALSE,
#     countsCol = 
#     
# )




