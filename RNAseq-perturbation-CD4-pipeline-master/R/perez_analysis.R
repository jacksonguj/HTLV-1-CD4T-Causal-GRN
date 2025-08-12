read_lupus_cells = function(path = "/oak/stanford/groups/pritch/users/jweinstk/resources/lupus_ye_perez/adata_T4_em_2023_05_26.csv") {
    df = vroom::vroom(
                  path,
                  col_select = c(
                        donor_id,
                        disease,
                        sex,
                        phase,
                        disease_state,
                        Processing_Cohort,
                        suspension_uuid,
                        matches("score")
                    )
    ) %>%
        dplyr::rename(
            cohort_batch = Processing_Cohort,
            suspension_batch = suspension_uuid
        ) %>%
        dplyr::mutate(
            dplyr::across(c(cohort_batch, suspension_batch), as.factor)
        ) %>%
        dplyr::mutate(
            disease = relevel(factor(disease), ref = "normal"),
            disease_state = relevel(factor(disease_state), ref = "na")
        )
    return(df)
}

lupus_covars = function() {

    covars = c(
        "sex",
        "S_score",
        "G2M_score",
        # "total_module_score",
        "cohort_batch"
    )

    covar_string = paste(covars, collapse = " + ")
    return(c(
      "full" = glue::glue("disease + {covar_string}"),
      "reduced" = glue::glue("{covar_string}")
    ))
}

lupus_state_covars = function() {

    covars = c(
        "sex",
        "S_score",
        "G2M_score",
        "total_module_score",
        "cohort_batch"
    )

    covar_string = paste(covars, collapse = " + ")
    return(c(
      "full" = glue::glue("disease_state + {covar_string}"),
      "reduced" = glue::glue("{covar_string}")
    ))
}

cell_cycle_covars = function() {

    covars = c(
        "sex",
        "cohort_batch"
    )

    covar_string = paste(covars, collapse = " + ")
    return(c(
      "full" = glue::glue("G2M_score + {covar_string}"),
      "reduced" = glue::glue("{covar_string}")
    ))
}

regress_module_signatures = function(lupus, gene_modules, covars = lupus_covars(), main_effect = "disease") {
    modules = names(gene_modules)
    score_labels = glue::glue("score_{modules}")
    models = tibble::tibble(
            modules = score_labels,
            model = purrr::map(modules, ~{
                    logger::log_info(glue::glue("Now working on {.x}"))
                    formula_full = as.formula(glue::glue("{.x} ~ {covars['full']} + (1|donor_id)"))
                    formula_reduced = as.formula(glue::glue("{.x} ~ {covars['reduced']} + (1|donor_id)"))
                    model_full = lme4::lmer(formula_full, lupus, REML = FALSE)
                    model_reduced = lme4::lmer(formula_reduced, lupus, REML = FALSE)
                    diff = 2 * (logLik(model_full) - logLik(model_reduced))
                    pvalue = pchisq(diff, df = 1, lower.tail = FALSE)
                    if(is.character(lupus[[main_effect]]) | is.factor(lupus[[main_effect]])) {

                        term_levels = as.character(levels(factor(lupus[[main_effect]])))
                        ref = term_levels[1]
                        term_levels = setdiff(term_levels, ref)
                        # term = "diseasesystemic lupus erythematosus"
                        term = glue::glue("{main_effect}{term_levels}")
                    } else {
                        term = glue::glue("{main_effect}")
                    }
                    print(glue::glue("term = {term}"))
                    beta = lme4::fixef(model_full)[term]
                    se = coef(summary(model_full))[term, "Std. Error"]
                    conf = confint(model_full, method = "Wald")[term, ]
                    ret = tibble::tibble(
                            "module" = rep(.x, length(term)),
                            "term" = term,
                            "2llr" = rep(diff, length(term)),
                            "estimate" = beta,
                            "std.error" = se,
                            "p.value" = pvalue
                        )
                    if (length(term) > 1) {
                        ret = ret %>%
                            dplyr::mutate(conf.low = conf[, 1], conf.high = conf[, 2])
                    } else {
                        ret = ret %>%
                            dplyr::mutate(conf.low = conf[1], conf.high = conf[2])
                    }
                    return(ret)
                }, .progress = TRUE)
        )

    result = models %>%
        dplyr::pull(model) %>%
        dplyr::bind_rows(.)

    return(result)
}

plot_module_signatures = function(signature_estimates, tag = "SLE") {

    plot = signature_estimates %>%
        dplyr::mutate(
            label = stringr::str_remove_all(module, "score_"),
            label = forcats::fct_reorder(label, estimate, .desc = FALSE)
        ) %>%
        ggplot(data = ., aes(y = label, x = estimate, xmin = conf.low, xmax = conf.high, colour = p.value < .05)) +
        geom_pointrange() +
        cowplot::theme_cowplot(font_size = 12) +
        labs(x = glue::glue("{stringr::str_replace(tag, '_', ' ')} effect on module expression\nin CD4+ FOXP3- T-cells (95% CI)"), y = "", colour = "pvalue < 0.05") +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        theme(axis.title.y = element_blank())

    ggsave(
        filename = file.path(figure_dir(), glue::glue("{tag}_module_forest_2023_04_27.pdf")),
        plot,
        width = 4,
        height = 4,
        units = "in"
    )
}

plot_module_disease_state = function(signature_estimates, tag = "SLE") {

    plot = signature_estimates %>%
        dplyr::mutate(
            label = stringr::str_remove_all(module, "score_"),
            label = forcats::fct_reorder(label, estimate, .desc = FALSE),
            facet_label = stringr::str_remove_all(term, "disease_state"),
            sig = 0 < conf.low | conf.high < 0
        ) %>%
        ggplot(data = ., aes(y = facet_label, x = estimate, xmin = conf.low, xmax = conf.high, colour = sig)) +
        geom_pointrange() +
        facet_grid(rows = vars(label)) + 
        cowplot::theme_cowplot(font_size = 12) +
        cowplot::panel_border() + 
        labs(x = glue::glue("{stringr::str_replace(tag, '_', ' ')} effect on module expression\nin CD4+ FOXP3- T-cells (95% CI)"), y = "", colour = "pvalue < 0.05") +
        geom_vline(xintercept = 0, color = "gray", linetype = "dashed", alpha = .7) +
        scale_color_manual(values = c("TRUE" = "#c23121", "FALSE" = "grey")) +
        theme(axis.title.y = element_blank())

    ggsave(
        filename = file.path(figure_dir(), glue::glue("{tag}_module_forest_2023_04_27.pdf")),
        plot,
        width = 4,
        height = 6,
        units = "in"
    )
}
