# This will run all the multilevel categorical logistic regression analyses.

# Below, I am using GNU Parallel. 
# The author insists that we cite this particular paper:
# O. Tange (2011). GNU Parallel - The Command-Line Power Tool. 
# ;login: The USENIX Magazine, 36, 42-47. 
# The --will-cite indicates that I will cite. And I just did cite. 

parallel --will-cite Rscript run_recall_model.R ::: k1_a k1_p k1_c k2_pa k2_pc k2_ca k3 null
