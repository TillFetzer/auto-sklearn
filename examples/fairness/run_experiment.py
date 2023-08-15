import base
import corrleation_remover
import redlineing
import lfr
import base_cr
import single_base
import sampling
import base_sampling
import base_sampling_cr
import base_sampling_cr_com
import base_lfr
import base_sampling_cr_lfr
import autosklearn
import single_lfr
import base_sampling_lfr
import base_unbalanced
import base_cr_lfr
import base_cr_lfr_com
import sampling_cr_lfr

import base_redlineing
import base_redlineing_cr
import base_redlineing_ps
import base_redlineing_lfr

import base_redlineing_cr_lfr
import base_redlineing_ps_lfr
import base_redlineing_ps_cr
import base_redlineing_ps_cr_lfr
import redlineing_ps_cr_lfr

import base_sar_ps_com
import complety_pipeline

def run_experiment(datasets =["adult"],
 fairness_constrains=["demographic_parity"],
  methods=["moo"], 
  file="/work/dlclarge2/fetzert-MySpace/results/", 
  seeds=[42,42,42],
   sf=["sex"] ,
   runtime = 108000, 
   runcount=-1,
   under_folder="dummy",
   performance = autosklearn.metrics.accuracy):
    # sf= ["foreign_worker"]
    print("start")
    if len(sf) == 1:
        sf = len(datasets) * sf
    for constrain in fairness_constrains:
        for i, dataset in enumerate(datasets):
            for seed in seeds:
                for method in methods:
                    if method == "moo_unbalanced":
                        base_unbalanced.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo":
                        base.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "redlineing":
                        redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "cr":
                        corrleation_remover.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "lfr":
                        lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "so_lfr":
                        single_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+cr":
                        base_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "so":
                        single_base.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "ps":
                        sampling.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "ps":
                        sampling.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+ps":
                        base_sampling.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+ps+cr":
                        base_sampling_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+ps*cr":
                        base_sampling_cr_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance) 
                    if method == "moo+lfr":
                        base_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+ps+cr+lfr":
                        base_sampling_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+ps+lfr":
                        base_sampling_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)        
                    if method == "moo+cr+lfr":
                        base_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)        
                    if method == "ps+cr+lfr":
                        sampling_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance) 
                    if method == "moo+cr*lfr":
                        base_cr_lfr_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    # experiemtns with redlineing
                    if method == "redlineing":
                        redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar":
                         base_redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+cr":
                        base_redlineing_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+ps":
                        base_redlineing_ps.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+lfr":
                        base_redlineing_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, 5, under_folder, performance)
                    
                    if method == "moo+sar+cr+lfr":
                        base_redlineing_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+ps+lfr":
                        base_redlineing_ps_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+ps+cr":
                        base_redlineing_ps_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar+ps+cr+lfr":
                        base_redlineing_ps_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "sar+ps+cr+lfr":
                        redlineing_ps_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
                    if method == "moo+sar*ps":
                        base_sar_ps_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)   
                    if method == "lfr_test":
                        lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, 5, under_folder, performance)#
                    if method == "cp":
                        complety_pipeline.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder, performance)
            print("all runs of {} finished".format(dataset))
    print("finished")
