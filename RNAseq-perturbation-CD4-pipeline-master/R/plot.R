suppressPackageStartupMessages({
    library(ggplot2)
    library(ggrepel)
    library(dplyr)
    library(cowplot)
    library(vroom)
    library(purrr)
    library(tidyr)
    library(glue)
    library(logger)
    library(igraph)
    library(tidygraph)
    library(ggraph)
})

source("utils.R")

# CONFIG ---------------------------
FILTER = TRUE
FILTER_VAR = "PIP"
FILTER_THRESHOLD = .50

input_dir = function() {
    glue::glue("{Sys.getenv('SCRATCH')}/InferCausalGraph/output")
}

csv_input_dir = file.path(input_dir(), "csv")
txt_input_dir = file.path(input_dir(), "txt")

meta = "0.01_2022_06_06_hyttinen_inspired_scaled"
filter_meta = "unfiltered"

gold_standard = tibble(
    row = c(
        "STAT5A",
        "STAT5B",
        "ETS1",
        "ETS1",
        "GATA3",
        "GATA3",
        "IRF4",
        "MYB"
    ),
    col = c(
        "IL2RA",
        "IL2RA",
        "IL2RA",
        "GATA3",
        "IL2RA",
        "ETS1",
        "IL2RA",
        "IL2RA"
    )
)

targets = readLines(file.path(txt_input_dir, "targets.txt"))

covs = vroom(file.path(csv_input_dir, glue("control_cov_{meta}.csv"))) %>%
    setNames(targets) %>% 
    as.matrix

total_effects = vroom(file.path(csv_input_dir, glue("total_effects_{meta}.csv")), col_names = FALSE) %>%
    setNames(targets) %>%
    mutate(row = targets) %>%
    pivot_longer(-row, names_to = "col", values_to = "total_effect")

constraint_matrix = vroom(file.path(csv_input_dir, glue("constraint_{meta}.csv")), col_names = FALSE)
experiment_outcome_vec = readLines(file.path(txt_input_dir, glue("outcome_experiment_vec_{meta}.txt"))) %>%
    as.numeric

parsed_chain_complete = vroom(file.path(csv_input_dir, glue("parsed_chain_{meta}.csv"))) 
parsed_chain = parsed_chain_complete

direct_effect_matrix = parsed_chain_complete %>%
    tidyr::complete(row, col, fill = list(estimate = 0)) %>%
    pivot_wider(names_from = "col", values_from = "estimate", row) %>%
    dplyr::slice(match(targets, row)) %>% #reorder rows
    dplyr::select(any_of(targets)) %>% #reorder cols
    as.matrix

indirect_effect_matrix = parsed_chain_complete %>%
    tidyr::complete(row, col, fill = list(estimate = 0)) %>%
    inner_join(total_effects) %>%
    mutate(indirect = total_effect - estimate) %>%
    pivot_wider(names_from = "col", values_from = "indirect", row) %>%
    dplyr::slice(match(targets, row)) %>% #reorder rows
    dplyr::select(any_of(targets)) %>% #reorder cols
    as.matrix

diag(direct_effect_matrix) = 0

total_effect_matrix = total_effects %>%
    pivot_wider(names_from = "col", values_from = "total_effect", row) %>%
    dplyr::slice(match(targets, row)) %>% #reorder rows
    dplyr::select(any_of(targets)) %>% #reorder cols
    as.matrix

log_info(glue::glue("Discovered {nrow(parsed_chain)} edges"))

if(FILTER) {
    parsed_chain = filter(parsed_chain, .data[[FILTER_VAR]] > .env[["FILTER_THRESHOLD"]])
    filter_meta = glue::glue("filtered_by_{FILTER_VAR}_{FILTER_THRESHOLD}")
    log_info(glue::glue("After filtering, retained {nrow(parsed_chain)} edges"))
}

genes = c("CBFB", "TNFAIP3", "KLF2", "FOXK1", "ATXN7L3", "ZNF217", "HIVEP2", "IRF2", "MYB", "IRF1", 
             "MED12", "YY1", "MBD2", "RELA", "ETS1", "PTEN", "FOXP1", "JAK3", "KMT2A", "IRF4", "GATA3", 
                          "STAT5A", "STAT5B", "IL2RA")

gene_grid = expand_grid(x = genes, y = genes) %>%
    left_join(
        parsed_chain %>% select(x = row, y = col, estimate),
        by = c("x", "y")
    ) %>%
    mutate(estimate = coalesce(estimate, 0))

heatmap = ggplot(data = gene_grid %>% mutate(x = factor(x, levels = genes), y = factor(y, levels = genes)),
    aes(x = y, y = x, fill = estimate)) +
    geom_tile() +
    cowplot::theme_cowplot(font_size = 12) +
    theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 0.95, hjust = 1)
    ) +
    scale_y_discrete(limits = rev) + 
    labs(fill = "edge") +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red")

ggsave(
    file.path(figure_output_dir(), glue("ggplot_directed_graph_matrix_{meta}_{filter_meta}.pdf")),
    heatmap,
    width = 6,
    height = 4,
    units = "in"
)

in_degree = parsed_chain %>%
    group_by(col) %>%
    summarize(in_degree = n()) %>%
    rename(gene = col)

out_degree = parsed_chain %>%
    group_by(row) %>%
    summarize(out_degree = n()) %>%
    rename(gene = row)

degree = full_join(in_degree, out_degree, by = "gene") %>%
    mutate(
        in_degree = coalesce(in_degree, 0),
        out_degree = coalesce(out_degree, 0)
    )

degree_plot = ggplot(data = degree, aes(x = out_degree, y = in_degree, label = gene)) +
    geom_point() +
    geom_text_repel() +     
    cowplot::theme_cowplot(font_size = 12) +
    labs(x = "Number of outgoing edges", y = "Number of incoming edges")

ggsave(
    file.path(figure_output_dir(), glue("ggplot_degree_{meta}_{filter_meta}.pdf")),
    degree_plot,
    width = 6,
    height = 4,
    units = "in"
)


total_direct_effect_plot = total_effects %>%
    dplyr::inner_join(parsed_chain) %>%
    mutate(
        label = glue::glue("{row}%->%{col}"),
        # parsed_label = map_chr(label, ~parse(text = as.expression(.x)))
        parsed_label = map_chr(label, ~as.character(as.expression(.x)))
    ) %>%
    ggplot(data = ., aes(x = estimate, y = total_effect, label = parsed_label, color = row)) +
        cowplot::theme_cowplot(font_size = 12) +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
        geom_point() + 
        geom_text_repel(max.overlaps = 5, parse = TRUE) +
        labs(x = "Direct effect", y = "Total effect", color = "")
        # geom_smooth(method = "lm") 

ggsave(
    file.path(figure_output_dir(), glue("ggplot_scatter_direct_total_{meta}_{filter_meta}.pdf")),
    total_direct_effect_plot,
    width = 6,
    height = 4,
    units = "in"
)

indirect_direct_effect_plot = total_effects %>%
    dplyr::inner_join(parsed_chain) %>%
    mutate(
        label = glue::glue("{row}%->%{col}"),
        indirect = total_effect - estimate,
        # parsed_label = map_chr(label, ~parse(text = as.expression(.x)))
        parsed_label = map_chr(label, ~as.character(as.expression(.x)))
    ) %>%
    ggplot(data = ., aes(x = estimate, y = indirect, label = parsed_label, color = row)) +
        cowplot::theme_cowplot(font_size = 12) +
        geom_point() + 
        geom_text_repel(max.overlaps = 5, parse = TRUE) +
        labs(x = "Direct effect", y = "Indirect effect", color = "") 
        # geom_smooth(method = "lm") 

ggsave(
    file.path(figure_output_dir(), glue("ggplot_scatter_direct_indirect_{meta}_{filter_meta}.pdf")),
    indirect_direct_effect_plot,
    width = 6,
    height = 4,
    units = "in"
)


create_tidy_graph = function(parsed_chain) {

    graph = tbl_graph(
        nodes = tibble(name = targets),
        edges = parsed_chain %>% 
            select(from = row, to = col, signed_weight = estimate) %>%
            mutate(weight = abs(signed_weight))
    )

    return(graph)
}

