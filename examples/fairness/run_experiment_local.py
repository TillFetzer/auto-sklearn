import base
import autosklearn
import corrleation_remover
import lfr
import base_lfr
import argparse
import redlineing
import base_cr
import single_base
import sampling
import base_sampling
import base_sampling_cr
import base_sampling_lfr
import base_cr_lfr
import sampling_cr_lfr
import base_sampling_cr_lfr
import base_sampling_cr_com
import base_cr_lfr_com
import base_sar_ps_com
import redlineing_ps_cr_lfr
import utils_fairlearn

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
import base_pipeline

parser=argparse.ArgumentParser()

parser.add_argument("--d", help="datsets",nargs="*")
parser.add_argument("--m", help="Foo the program",nargs="*", default=[])
parser.add_argument("--r", help="Foo the program", type = int)
parser.add_argument("--fc", help="Foo the program", nargs="*", default=[])
parser.add_argument("--sa", help="Foo the program", nargs="*",default=[])
parser.add_argument("--seeds", help="Foo the program", type=int, nargs="*", default=[])
parser.add_argument("--runcount", help="Foo the program", type=int)
parser.add_argument("--f")
args=parser.parse_args()

datasets = args.d
fairness_constrains = args.fc
runtime = args.r
methods = args.m
file = args.f
sf = args.sa # same length then dataset or 1
seeds = args.seeds
runcount = args.runcount
performance = utils_fairlearn.set_f1_score()
# sf= ["foreign_worker"]
print("start")
if len(sf) == 1:
    sf = len(datasets) * sf
for constrain in fairness_constrains:
    for i, dataset in enumerate(datasets):
        for seed in seeds:
            for method in methods:
                if method == "moo":
                    base.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test", performance = performance)
                if method == "sar":
                    redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "cr":
                    corrleation_remover.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test",performance = autosklearn.metrics.f1_macro)
                if method == "lfr":
                   lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+lfr":
                    base_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+cr":
                    base_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test", performance = autosklearn.metrics.f1_macro)
                if method == "so":
                    single_base.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "ps":
                    sampling.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+ps":
                    base_sampling.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+ps+cr":
                    base_sampling_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+ps+cr+lfr":
                    base_sampling_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+ps*cr":
                    base_sampling_cr_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test") 
                if method == "moo+cr*lfr":
                    base_cr_lfr_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "moo+ps+lfr":
                    base_sampling_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")        
                if method == "moo+cr+lfr":
                    base_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")        
                if method == "ps+cr+lfr":
                    sampling_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")   
                if method == "moo+sar*ps":
                    base_sar_ps_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")   
                if method == "sar+ps+cr+lfr":
                    redlineing_ps_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder="test")
                if method == "redlineing":
                    redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar":
                    base_redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+cr":
                    base_redlineing_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+ps":
                    base_redlineing_ps.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+lfr":
                    base_redlineing_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, 5, under_folder = "test")    
                if method == "moo+sar+cr+lfr":
                    base_redlineing_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+ps+lfr":
                    base_redlineing_ps_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+ps+cr":
                    base_redlineing_ps_cr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar+ps+cr+lfr":
                    base_redlineing_ps_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "sar+ps+cr+lfr":
                    redlineing_ps_cr_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo+sar*ps":
                    base_sar_ps_com.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")   
                if method == "lfr_test":
                    lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, 5, under_folder = "test")#
                if method == "cp":
                    complety_pipeline.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "moo_sar_lfr":
                    base_redlineing_lfr.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
                if method == "bp":
                    base_pipeline.run_experiment(dataset, constrain, sf[i], runtime, file, seed, runcount, under_folder = "test")
print("finished")
