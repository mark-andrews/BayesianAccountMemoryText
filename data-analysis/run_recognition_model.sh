# We experimented with a set of models that differed in their number of predictors.
# The "full" model, i.e. with all predictors and all interactions and all random effects that are theoretically justified, is v6.
# Models listed v1 to v5 have dropped some random effects and some interactions.

# To recreate the results reported in the final manuscript, the following three commands need to be run. 
# Each one executes multiple R scripts in parallel, and each R script itself spawned multiple child processes (Stan models).
# All three eventually use up either more or less all 36 Intel Xeon Gold 6154 cores or more or less all the 250GB of ram. 
# This is why they need to run on their own.

parallel Rscript run_recognition_model.R v6 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null 
parallel Rscript get_model_loowaic.R v6 ::: k3 k2_cp k2_ca k2_pa
parallel Rscript get_model_loowaic.R v6 ::: k1_p k1_a k1_c null

# All the commands to run the other model versions (v1 to v5) are here for reference.

# parallel Rscript run_recognition_model.R v1 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
# parallel Rscript run_recognition_model.R v2 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
# parallel Rscript run_recognition_model.R v3 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
# parallel Rscript run_recognition_model.R v4 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
# parallel Rscript run_recognition_model.R v5 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
# parallel Rscript run_recognition_model.R v6 ::: k3 k2_ca k2_cp k2_pa k1_p k1_c k1_a null
 
# parallel Rscript get_model_loowaic.R v6 ::: k3 k2_cp k2_ca k2_pa
# parallel Rscript get_model_loowaic.R v5 ::: k3 k2_cp k2_ca k2_pa
# parallel Rscript get_model_loowaic.R v4 ::: k3 k2_cp k2_ca k2_pa
# parallel Rscript get_model_loowaic.R v3 ::: k3 k2_cp k2_ca k2_pa
# parallel Rscript get_model_loowaic.R v2 ::: k3 k2_cp k2_ca k2_pa
# parallel Rscript get_model_loowaic.R v1 ::: k3 k2_cp k2_ca k2_pa
 
# parallel Rscript get_model_loowaic.R v1 ::: k1_p k1_a k1_c null
# parallel Rscript get_model_loowaic.R v2 ::: k1_p k1_a k1_c null
# parallel Rscript get_model_loowaic.R v3 ::: k1_p k1_a k1_c null
# parallel Rscript get_model_loowaic.R v4 ::: k1_p k1_a k1_c null
# parallel Rscript get_model_loowaic.R v5 ::: k1_p k1_a k1_c null
# parallel Rscript get_model_loowaic.R v6 ::: k1_p k1_a k1_c null
