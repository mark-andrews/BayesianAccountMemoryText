library(readr)
library(lme4)
library(MuMIn)
library(dplyr)
library(tibble)
library(tidyr)
library(magrittr)
library(stringr)
library(brms)
library(assertthat)
library(latex2exp)
library(ggpubr)
library(cowplot)
library(purrr)

clip_tikz_bounding_box <- function(filename){

  # tikzdevice adds too much white space around its figure
  # this clips the white space
  # Thanks to 
  # https://stackoverflow.com/a/41186942/1009979
    
  lines <- readLines(con=filename)
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=filename)

}

re_scale <- function(x) {
  
  # Normalize (mean=0, sd=1) the log of x
  as.numeric(scale(x))
  
}

rescale_and_log <- function(x) {
  
  # Normalize (mean=0, sd=1) the log of x
  
  re_scale(log(x))
  
}

map_booleans <- function(x) {
  
  # Convert the strings 'False', 'True' to Booleans FALSE and TRUE
  
  as.logical(plyr::mapvalues(x, c('False', 'True'), c(F, T), warn_missing = F))
  
}

check_predictors <- function(Df){
  
  # Ensure we have no 0's or NA's in our main predictor variables
  # Raise error if so.
  
  all(sapply(Df[,c('cooccurrence.predictions', 'posterior.predictions', 'association.predictions')], 
             function(x) min(x) > 0.0 & is.finite(x)))
  
}

get_recognition_data <- function(recognitiondata_csvfile){
  
  # Minimal cleaning and processing of the recognition memory data
  
  Df <- read_csv(recognitiondata_csvfile) %>% 
    select(-X1) %>%   # Drop "X1" variable (column with row index)
    mutate(expected = map_booleans(expected),  # Convert 'True', 'False' to T, F 
           response = map_booleans(response),
           hit = map_booleans(hit),
           correct = map_booleans(correct)) %>% 
    filter(word != 'dhow') # Filter out the rows with word = 'dhow' (caused by error downstream)
  
  check_predictors(Df)
  
  Df %>% mutate(posterior.predictions = rescale_and_log(posterior.predictions),
                association.predictions = rescale_and_log(association.predictions),
                cooccurrence.predictions = rescale_and_log(cooccurrence.predictions)
  )
  
}

get_recall_data <- function(recalldata_csvfile){
  
  # Minimal cleaning and processing of the recall memory data
  
  Df <- read_csv(recalldata_csvfile) %>% 
    select(-X1) %>%   # Drop "X1" variable (column with row index)
    mutate(accuracy = map_booleans(accuracy)) # Convert 'True', 'False' to T, F 
  
  check_predictors(Df)
  
  Df %>% rename(posterior_predictions = posterior.predictions,
                association_predictions = association.predictions,
                cooccurrence_predictions = cooccurrence.predictions
  ) %>% mutate(posterior_predictions_ln = rescale_and_log(posterior_predictions),
               association_predictions_ln = rescale_and_log(association_predictions),
               cooccurrence_predictions_ln = rescale_and_log(cooccurrence_predictions)
  )
  
}


get_behavioural_experiment_summary_stats <- function(Df_recognition, 
                                                     Df_recall){
  
  # Return a list, see below, with some useful summary stats
  # from the behavioural experiment
  
  percentile_range_lims <- function(x, p=0.95, k=1){
    
    # A convenience function
    # Get the 95% interval range, upper and lower lims
    # Return either upper or lower lim, not both
    
    q <- (1 - p)/2 
    quants <- quantile(x, probs = c(q, 1-q))
    quants[k] %>% unname()
    
  }
  
  # An ad-hoc data frame for the calculation of subject totals, etc.
  Df_session <- rbind(select(Df_recognition, session, subject, age, sex, slide),
                      select(Df_recall, session, subject, age, sex, slide)) %>% 
    distinct()
  
  within(list(),{
    # How many memory tests; 
    # How many participants;
    # Gender breakdown, age;
    # Recognition memory hit rate, 
    # accuracy, false positive rate, etc.
    
    n_recognition_sessions <- length(unique(Df_recognition$slide))
    n_recall_sessions <- length(unique(Df_recall$slide))
    n_sessions <- length(unique(Df_session$slide))
    
    n_recognition_subjects <- length(unique(Df_recognition$subject))
    n_recall_subjects <- length(unique(Df_recall$subject))
    n_subjects <- length(unique(Df_session$subject))
    
    # Do some checks
    assert_that(n_recognition_sessions +  n_recall_sessions == n_sessions);
    
    gender_counts <- Df_session %>% 
      select(subject, sex) %>% 
      distinct() %>% 
      group_by(sex) %>% 
      summarize(count = n()) %>% 
      spread(sex, count) %>% 
      unlist()
    
    sessions_per_subject <- Df_session %>% 
      group_by(subject) %>% 
      summarise(count = n()) %>% 
      select(count) %>% 
      summarize(mean = mean(count)) %>% unlist() %>% unname()
    
    
    age_stats <- Df_session %>% 
      distinct(subject, age) %>% 
      summarize(median = median(age), 
                lower = percentile_range_lims(age, k=1), 
                upper = percentile_range_lims(age, k=2)) %>% unlist()
    
    recognition_test_list_length <- 20 # I just know that there are 20 words per list
  
    # NB: The 20 items per recognition test can be confirmed(-ish).
    # From this, we know that there can be no less than 20 items
    # per slides.
    assert_that(Df_recognition %>% 
                  group_by(slide) %>%
                  summarize(hits = n()) %>% 
                  summarise(max_items_per_slides = max(hits)) %>%
                  unlist() %>% 
                  unname() == recognition_test_list_length
    )
    
    number_of_recognition_tests <- 50 # again, we just know
    
    # But we can also verify by
    assert_that(max(Df_recognition$text) == number_of_recognition_tests)
    
    
    
    # Recognition memory test summary stats
    hit_rate <- Df_recognition %>% 
      group_by(slide) %>%
      summarize(hits = n()) %>% 
      summarize(hit_rate = mean(hits)/recognition_test_list_length) %>% 
      unlist()
    
    accuracy_rate <- Df_recognition %>%
      group_by(slide) %>% 
      summarize(accuracy_per_slide = mean(correct)) %>%
      summarize(average_accuracy = mean(accuracy_per_slide)) %>% 
      unlist() %>% 
      unname()
    
    Df_recognition_errors <- Df_recognition %>% 
      group_by(slide, expected) %>%
      summarize(response = mean(response)) %>% 
      ungroup() %>% 
      spread(expected, response) %>% 
      mutate(false_positive_rate = `FALSE`,
             false_negative_rate = 1 - `TRUE`) %>%
      select(slide, false_positive_rate, false_negative_rate)
    
    get_recognition_type_rates <- function(Df, recognition_type){
      (Df %>% 
        summarise(false_positive_rate = mean(false_positive_rate, na.rm = T),
                  false_negative_rate = mean(false_negative_rate, na.rm = T)
        ) %>% unlist())[recognition_type]
    }
    
    get_recognition_type_quantiles <- function(Df, recognition_type, p=0.1){
      (Df %>% 
         summarise(false_positive_rate = mean(false_positive_rate >= p),
                   false_negative_rate = mean(false_negative_rate >= p))
       %>% unlist())[recognition_type] %>% unname()
    }
    
    # When test word was *not* in the text, what was the
    # rate of positive responses?
    false_positive_rate <- get_recognition_type_rates(Df_recognition_errors, 'false_positive_rate')
    
    # When the test word was in the text, what was the
    # rate of negative responses?
    false_negative_rate <- get_recognition_type_rates(Df_recognition_errors, 'false_negative_rate')
 
 
    # On what proportion of tests was the false-positive/false-negative rate above 10%
    false_positive_rate_gt_10pc <- get_recognition_type_quantiles(Df_recognition_errors, 'false_positive_rate', p=0.1)
    false_negative_rate_gt_10pc <- get_recognition_type_quantiles(Df_recognition_errors, 'false_negative_rate', p=0.1)
    
    
    # Recall rates
    
    # On average how many words were recalled per subject
    # On average how many true and false recalls
    recall_rates <- Df_recall %>%
      group_by(slide) %>%
      summarize(n_recall = length(word),
                n_true_recall = sum(accuracy),
                n_false_recall = sum(!accuracy),
                false_recall_rate = n_false_recall/n_recall) %>%
      summarize(n = mean(n_recall),
                true_recall = mean(n_true_recall),
                false_recall = mean(n_false_recall),
                false_recall_rate = mean(false_recall_rate)) %>% 
      unlist()
    
    false_positive_recall_rate_greater_than_10pc <- Df_recall %>%
      group_by(slide) %>%
      summarise(false_recall = (sum(!accuracy)/length(word)) > 0.1) %>% 
      summarise(mean(false_recall))
    
  })
}

get_jags_waic_results <- function(cache_directory){
  
  file_names <- list.files(cache_directory, 
                           full.names = T, 
                           pattern = 'recall_save_')
  
  get_modelname <- function(filename){
    str_split(filename, pattern = '/')[[1]][3] %>% 
      str_split(pattern = '\\.')[[1]][1] %>% 
      str_split(pattern = 'recall_save_')[[1]][2]
  }
  
}

get_waic_results <- function(cache_directory){
  
  #model_filename_pattern <- 'stan_recognition_model__v[1-6]__(k3|k2_cp|k2_pa|k2_ca|k1_p|k1_c|k1_a|null)__seed_[[:digit:]]+__chains_[[:digit:]]__warmup_[[:digit:]]+__iterations_[[:digit:]]+__samples_[[:digit:]]+\\.Rds'
  model_loowaic_filename_pattern <- 'stan_recognition_model__v[1-6]__(k3|k2_cp|k2_pa|k2_ca|k1_p|k1_c|k1_a|null)__seed_[[:digit:]]+__chains_[[:digit:]]__warmup_[[:digit:]]+__iterations_[[:digit:]]+__samples_[[:digit:]]+__loowaic\\.Rds'
  
  get_stanfilenames <- function(cache_directory, filename){
    
    get_modelname <- function(filename){
      filename_split <- stringr::str_split(filename, pattern = '__')[[1]]
      paste(filename_split[2], filename_split[3], sep = '__')
    }
    
    stanfilenames <- list.files(cache_directory,
                                full.names = T,
                                pattern = filename)
    
    names(stanfilenames) <- lapply(stanfilenames, get_modelname) %>% unlist()
    
    stanfilenames
  }
  
  # get_stanmodel_filenames <- function(cache_directory){
  #   get_stanfilenames(cache_directory, model_filename_pattern)
  # }
  
  get_stanmodel_loowaic_filenames <- function(cache_directory){
    get_stanfilenames(cache_directory, model_loowaic_filename_pattern)
  }
  
  lapply(get_stanmodel_loowaic_filenames(cache_directory), 
         readRDS)
  
}

get_waic_results_summary <- function(cache_directory, version='v6'){
  
  M <- get_waic_results(cache_directory)
  
  
  y <- do.call(compare_ic,
               lapply(M[names(M)[str_detect(names(M), version)]], 
                      function(x) x[['M_waic']])
  )
  
  within(list(), {
    
    waic <- do.call(rbind, 
                    lapply(y[str_detect(names(y), sprintf('%s_', version))], 
                           function(x) x$estimates['waic',])
    ) %>% 
      as.data.frame() %>%
      rownames_to_column('model') %>% 
      separate(model, into=c('v','model'), sep='__') %>% 
      select(model, waic=Estimate)
    
    
    
    dwaic <- y$ic_diffs__ %>%
      as.data.frame() %>%
      rownames_to_column('comparison') %>% 
      rename(dwaic = WAIC,
             se = SE) %>%
      arrange(dwaic) %>% 
      mutate(comparison = str_replace_all(comparison, sprintf('%s__', version), ''),
             version = version) %>%
      separate(comparison, into=c('left', 'right'), sep=' - ', remove=F) %>% 
      select(comparison, left, right, dwaic, se)
  })
  
}

waic_select <- function(waic_results_summary, model_name){
  waic_results_summary$waic %>% 
    filter(model == model_name) %>% 
    select(waic) %>% 
    unlist() %>% 
    unname()
}

waic_select_as_chr <- function(waic_results_summary, model_name){
  waic_select(waic_results_summary, model_name) %>%
	      round(digits = 2) %>%
	      as.character()
}

dwaic_select <- function(waic_results_summary, model_a, model_b){
  
  dwaic_a <- waic_select(waic_results_summary, model_b) - 
    waic_select(waic_results_summary, model_a)
  
  
  results <- waic_results_summary$dwaic %>% 
    filter(str_detect(comparison, model_a)) %>% 
    filter(str_detect(comparison, model_b)) %>% 
    select(dwaic, se) %>% 
    unlist()
  
  # dwaic_a and results['dwaic'] ought to be identical
  # but one could be the negative of the other
  # as a consequence of how the info is laid out
  # in the tables
  # Check if they are equal. if not, check if
  # one is the neg of the other, if so
  # then reverse results['dwaic'].
  # If neither of these two scenarios, then raise error.
  
  dwaic_b <- results['dwaic'] %>% unname()
  
  equality_test <- function(x, y, tol=3){
    round(x, digits = tol) == round(y, digits = tol)
  }
  
  if (equality_test(dwaic_a, dwaic_b)){
    return(results)
  } else if (equality_test(dwaic_a, -dwaic_b)){
    results['dwaic'] <- -results['dwaic']
    return(results)
  } else {
    stop(sprintf('dwaic failed check: %2.3f, %2.2f', dwaic_a, dwaic_b))
  }

}

dwaic_select_as_chr <- function(waic_results_summary, model_a, model_b){
  results <- dwaic_select(waic_results_summary, model_a, model_b)
  sprintf('%2.2f (%2.2f)', results[1], results[2])
}

get_recognition_preliminary_results <- function(Df_recognition, eps=0.1){
  
  logit <- function(p, eps=0.1) {
    
    # log odds of probability p
    # threshold values less than or greater than eps, and 1-eps
    # to avoid Inf/-Inf in log odds transform
    
    p <- ifelse(p > 1-eps, 1-eps, ifelse(p < eps, eps, p))
    p <- log(p/(1-p)) # log odds
  }
  
  within(list(),{
    eps <- 0.1 # To avoid Inf/-Inf in logit transform
    Df <- Df_recognition %>% 
      mutate(bayes = posterior.predictions,
             cooccur = cooccurrence.predictions,
             assoc = association.predictions) %>% 
      group_by(stimulus, expected) %>% 
      summarise(response = mean(response, na.rm=T),
                logit_response = logit(response, eps),
                bayes = mean(bayes, na.rm=T),
                cooccur = mean(cooccur, na.rm=T),
                assoc = mean(assoc, na.rm=T),
                text = mean(text)) %>% 
      ungroup()
    
    model_list <- list(bayes='bayes',
                       cooccur='cooccur',
                       assoc='assoc')
    
    Df_rsquared <- do.call(cbind,
            lapply(model_list,
                   function(predictor){
                     text_ids <- unique(Df$text)
                     r_squared <- sapply(text_ids, 
                                         function(text_id){
                                           formula_string <- sprintf('response ~ %s + expected',
                                                                     predictor)
                                           summary(lm(as.formula(formula_string), 
                                                      data=filter(Df, text==text_id)))$r.squared
                                         }
                     )
                     names(r_squared) <- text_ids
                     r_squared
                   })
    ) %>% 
      as.data.frame() %>% 
      rownames_to_column('text') %>% 
      as_tibble() %>% 
      mutate(text = as.numeric(text)) %>% 
      arrange(text) %>% 
      gather(model, r_squared, bayes:assoc) %>% 
      mutate(model = factor(model, c('bayes', 'cooccur', 'assoc')))
    
    Df_rsquared_median <- Df_rsquared %>% 
      group_by(model) %>% 
      summarize(median = median(r_squared)) %>% 
      spread(model, median) %>% 
      unlist()
    
    Df_rsquared_which_median <- lapply(model_list,
                                       function(var){
                                         Df_rsquared %>%
                                           filter(model==var) %>%
                                           mutate(d = abs(r_squared - Df_rsquared_median['bayes']),
                                                  is_min = d == min(d)) %>% 
                                           filter(is_min) %>% 
                                           select(text) %>% 
                                           unlist() %>% 
                                           unname()
                                       }) %>% unlist()
    
    # Spearman's rho correlation tests
    Df_spearman <- (function(Df){
      cor_test <- function(var_1, var_2){
        cor.test(var_1, var_2, method = 'spearman', exact=FALSE)$estimate
      }
      
      get_bayes_cor <- function(Df){
        Df %$% cor_test(response,bayes)
      }
      
      get_cooccur_cor <- function(Df){
        Df %$% cor_test(response,cooccur)
      }
      
      get_assoc_cor <- function(Df){
        Df %$% cor_test(response,assoc)
      }
      
      Df %>% 
        group_by(text, expected) %>% 
        do(bayes = get_bayes_cor(.),
           cooccur = get_cooccur_cor(.),
           assoc = get_assoc_cor(.)) %>% 
        unnest() %>% 
        gather(model, correlation, bayes:assoc) %>% 
        na.omit() %>% 
        mutate(model = factor(model, c('bayes', 'cooccur', 'assoc')))
      
    })(Df)
    
    rho_median <- Df_spearman %>% 
      group_by(model) %>% 
      summarize(median=median(correlation)) %>% 
      spread(model, median) %>% 
      unlist()
    
    # This stuff is for the chop I think
    # ---------------------------
    # Model comparison stuff
    # M <- lapply(list(k3 = 'logit_response ~ bayes + cooccur + assoc + expected + (1|text)',
    #                  k2_bc = 'logit_response ~ bayes + cooccur + expected + (1|text)',
    #                  k2_ba = 'logit_response ~ bayes + assoc + expected + (1|text)',
    #                  k2_ca = 'logit_response ~ cooccur + assoc + expected + (1|text)',
    #                  k1_b = 'logit_response ~ bayes + expected + (1|text)',
    #                  k1_c = 'logit_response ~ cooccur + expected + (1|text)',
    #                  k1_a = 'logit_response ~ assoc + expected + (1|text)'),
    #             function(formula) lmer(as.formula(formula), REML=F, data=Df)
    # )
    
    # model_comparison_table <- (function(M){
    #   test_drop_cooccur <- anova(M$k3, M$k2_ba)
    #   test_drop_assoc <- anova(M$k3, M$k2_bc)
    #   test_drop_bayes <- anova(M$k3, M$k2_ca)
    #   
    #   ll_full <- logLik(M$k3) %>% as.numeric()
    #   ll_drop_assoc <- logLik(M$k2_bc) %>% as.numeric()
    #   ll_drop_cooccur <- logLik(M$k2_ba) %>% as.numeric()
    #   ll_drop_bayes <- logLik(M$k2_ca) %>% as.numeric()
    #   
    #   deviance <- c(ll_full, ll_drop_bayes, ll_drop_cooccur, ll_drop_assoc) * -2
    #   
    #   pseudo_rsq_all <- lapply(M, 
    #                            function(M) r.squaredGLMM(M)[1,'R2m'] %>% unname()
    #   )
    #   
    #   pseudo_rsq <- c(pseudo_rsq_all$k3,
    #                   pseudo_rsq_all$k2_ca,
    #                   pseudo_rsq_all$k2_ba,
    #                   pseudo_rsq_all$k2_bc)
    #   
    #   
    #   get_test_stats <- function(test_results){
    #     chisq_df <- test_results$`Chi Df`[2]
    #     chisq <- test_results$Chisq[2]
    #     test_stats_string <- sprintf('$%2.2f$', chisq)
    #     
    #     p_value <- test_results$`Pr(>Chisq)`[2]
    #     if (p_value < 1e-10) {
    #       p_value_string <- '$(p \\ll 0.001)$'
    #     } else if (p_value < 0.001){
    #       p_value_string <- '$(p < 0.001)$'
    #     } else if (p_value < 0.01){
    #       p_value_string <- '$(p < 0.01)$'
    #     } else {
    #       p_value_string <- sprintf('$(p = %2.2f)$', p_value)
    #     } 
    #     
    #     paste(test_stats_string, p_value_string)
    #   }
    #   
    #   X <- rbind(sapply(pseudo_rsq, function(s) sprintf('%2.3f', s)), 
    #              sapply(deviance, function(s) sprintf('%2.3f', s)),
    #              c('0.00',
    #                get_test_stats(test_drop_bayes),
    #                get_test_stats(test_drop_cooccur),
    #                get_test_stats(test_drop_assoc))
    #   ) 
    #   
    #   colnames(X) <- c('Full model',
    #                    '$\\neg \\psibayes$',
    #                    '$\\neg \\psicooccur$',
    #                    '$\\neg \\psiassoc$')
    #   
    #   rownames(X) <- c('pseudo-$R^2$',
    #                    'Deviance',
    #                    '$\\Delta^{\\text{Deviance}}$')
    #   
    #   X
    # })(M)
    
    # get_prototypical_scatterplot <- function(predictor){
    #   text_id <- Df_rsquared_which_median[predictor]
    #   Df %>% 
    #     gather(model, phi, bayes:assoc) %>% 
    #     filter(text == text_id, model == predictor) %>% 
    #     ggplot(mapping = aes(x = phi, y = response, col = expected)) +
    #     geom_point() +
    #     xlim(-2, 2) + 
    #     ylim(0, 1) +
    #     stat_smooth(method='lm', se=F, size=0.5) +
    #     theme_classic() +
    #     scale_color_brewer(palette='Dark2') +
    #     coord_fixed(ratio = 4) +
    #     theme(legend.position="none")
    # }
    
    
    # --------------------------
    
    prototypical_scatterplots_data <- (function(Df){
      Df %>% 
        gather(model, phi, bayes:assoc) %>% 
        mutate(model = factor(model, c('bayes', 'cooccur', 'assoc'))) %>% 
        filter(
          (text == Df_rsquared_which_median['bayes'] & model == 'bayes') | 
            (text == Df_rsquared_which_median['cooccur'] & model == 'cooccur') | 
            (text == Df_rsquared_which_median['assoc'] & model == 'assoc')
        )
    })(Df)
  })
}

get_recall_preliminary_results <- function(Df_recall){
  
  # Log of a product is a sum of the logs
  sumlog <- function(x) sum(log(x))
  
  loglike_per_test_spread <- Df_recall %>%
    split(.$slide) %>% 
    map(function(Df) summarise_at(Df, vars(ends_with('_predictions')), funs(sumlog))) %>% 
    do.call(rbind, .) %>% 
    rename(bayes = posterior_predictions, 
           assoc = association_predictions, 
           cooccur = cooccurrence_predictions)
  
  loglike_per_test_diff_spread <- loglike_per_test_spread %>% 
    transmute(bayes_diff_cooccur = bayes - cooccur, 
              bayes_diff_assoc = bayes - assoc) 
 
  within(list(), {
  
    loglike_per_test <- loglike_per_test_spread %>% 
      gather(psi, log_likelihood)
    
    ll_overall <- loglike_per_test %>%
      group_by(psi) %>% 
      summarise(ll = sum(log_likelihood)) %>% 
      spread(psi,ll) %>% 
      unlist()
    
   loglike_per_test_diff_gt_0 <- loglike_per_test_diff_spread %>% 
      summarize(bayes_beats_cooccur = mean(bayes_diff_cooccur > 0),
                bayes_beats_assoc = mean(bayes_diff_assoc > 0)) %>%
      unlist()
    
    bf_positive_threshold <- log(3.0)
    loglike_per_test_diff_gt_3 <- loglike_per_test_diff_spread %>% 
      summarize(bayes_beats_cooccur = mean(bayes_diff_cooccur > bf_positive_threshold),
                bayes_beats_assoc = mean(bayes_diff_assoc > bf_positive_threshold)) %>%
      unlist()
    
    loglike_per_test_diff <- loglike_per_test_diff_spread %>% gather(psi, logbf)
    
    # median_probability_rank_per_test <- Df_recall %>%
    #   split(.$slide) %>% 
    #   map(function(Df) summarise_at(Df, vars(ends_with('_ranks')), funs(median))) %>% 
    #   do.call(rbind, .) %>% 
    #   rename(bayes = posterior_predictions_ranks, 
    #          assoc = association_predictions_ranks, 
    #          cooccur = cooccurrence_predictions_ranks) %>% 
    #   gather(psi, median_rank)
    
    ylim_lb <- -150 # this is used in the log likelihood plot
    
    loglikelihood_summaries <- loglike_per_test %>% 
      group_by(psi) %>% 
      summarize(median = median(log_likelihood),
                below_bound = sum(log_likelihood < ylim_lb)
      )
    
    loglikelihood_median <- loglikelihood_summaries %>% select(psi, median) %>% spread(psi, median) %>% unlist()
    loglikelihood_below_bound <- loglikelihood_summaries %>% select(psi, below_bound) %>% spread(psi, below_bound) %>% unlist()
    
  })

}

xtable_to_file <- function(the_table, filename){
  xtable(the_table) %>% 
    print.xtable(floating=FALSE, 
                 table.placement = NULL,
                 file = filename,
                 sanitize.text.function=function(x){x})
}

get_dwaic_jags <- function(model_A, model_B){
  
  # Using samples from two Jags models
  # return the difference of waic and its standard error
  
  get_pointwise_elpd <- function(x) log(apply(x, 2, mean))
  get_dse <- function(d) sqrt(length(d))*var(d)
  
  model_A_pointwise_elpd <- model_A$density_samples %>% get_pointwise_elpd()
  model_B_pointwise_elpd <- model_B$density_samples %>% get_pointwise_elpd()
  
  d_pointwise_elpd <- model_A_pointwise_elpd - model_B_pointwise_elpd
  
  
  c(dwaic = model_A$waic - model_B$waic, 
    dwaic_se = get_dse(d_pointwise_elpd))
  
}


get_jags_recall_models <- function(cache_directory){
  
  model_details_as_df <- list.files(cache_directory, 
                                    pattern='recall_save', 
                                    full.names = T) %>% 
    str_match('.*(recall_save_(.*)\\.Rds)') %>% 
    as_tibble() %>% 
    rename(path = V1, filename = V2, model_id = V3)
  
  model_details <- setNames(as.list(model_details_as_df$path), 
                            model_details_as_df$model_id)
  
  lapply(model_details, readRDS)
  
}

get_jags_recall_model_dwaic <- function(recall_models){
  
  lapply(recall_models, 
         function(model_a){
           lapply(recall_models, function(model_b){get_dwaic_jags(model_a, model_b)})
         }
  )
  
}

get_elpd_waic_i <- function(density_samples){
  # Following Section 4.1
  # WAIC and cross-validation in Stan
  # Aki Vehtari & Andrew Gelman 
  # preprint, May 31, 2014
  # http://www.stat.columbia.edu/~gelman/research/unpublished/waic_stan.pdf
  
  # This will be a vector of length n
  # where n is the number of observed data points
  elpd_waic_i <- log(apply(density_samples, 2, mean)) -
    apply(log(density_samples), 2, var)
  
  # We multiply by -2 for the deviance scale
  -2 * elpd_waic_i
}

get_elpd_waic <- function(density_samples){

  # What Vehtari et al call elpd_waic is the sum of 
  # the n independent terms
  # and the standard error is the sqrt of n 
  # times the variance 
  
  elpd_waic_i <- get_elpd_waic_i(density_samples)
  
  c(waic = sum(elpd_waic_i),
    waic_se = sqrt(length(elpd_waic_i) * var(elpd_waic_i))
  )
}


get_elpd_dwaic <- function(density_samples_A, density_samples_B){
  
  # A hopefully correct 
  
  
  elpd_waic_i_model_A <- get_elpd_waic_i(density_samples_A)
  elpd_waic_i_model_B <- get_elpd_waic_i(density_samples_B)
  
  d_elpd_waic_i <- elpd_waic_i_model_A - elpd_waic_i_model_B
  
  c(dwaic = sum(d_elpd_waic_i), # this *should* be the same as the difference of the waic of A and B
    dwaic_se = sqrt(length(d_elpd_waic_i) * var(d_elpd_waic_i))
  )

}

# Another method
get_jags_recall_model_dwaic_v2 <- function(recall_models){
  
  lapply(recall_models, 
         function(model_a){
           lapply(recall_models, function(model_b){get_elpd_dwaic(model_a$density_samples, 
                                                                  model_b$density_samples)})
         }
  )
  
}

# Another method
get_jags_recall_model_waic_v2 <- function(recall_models){
  lapply(recall_models, function(model){get_elpd_waic(model$density_samples)})
}
