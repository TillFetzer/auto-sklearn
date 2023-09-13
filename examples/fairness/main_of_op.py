from run_experiment import run_experiment
import argparse, sys
import autosklearn
import utils_fairlearn
parser=argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
parser.add_argument("--uf", type=str)
args=parser.parse_args()

idx = args.idx
under_folder = args.uf
seeds = [12345,25,42,45451,97]          
methods = ["moo+cr", "moo+sar+ps+cr+lfr"]
#methods = ["moo+sar+cr"]
datasets = ["german"]
#datasets = ["german"]
sfs = ["personal_status"]
#sfs = ["personal_status"]
fairness_constrains=["consistency_score","demographic_parity","equalized_odds", "error_rate_difference"]
#performance = utils_fairlearn.set_f1_score()
performance = autosklearn.metrics.accuracy
dataset = datasets[int(idx/(len(seeds)*len(methods)))%len(datasets)]
sf = sfs[int(idx/(len(seeds)*len(methods)))%len(datasets)]
method = methods[int(idx/len(seeds))%len(methods)]
seed = seeds[idx%len(seeds)]
fairness_constrains = fairness_constrains[int(idx/(len(seeds)*len(methods)*len(datasets)))]

#runcount = 10
runcount=200

print(dataset)
print(method)
print(seed)
print(fairness_constrains)
run_experiment(datasets =[dataset], 
               fairness_constrains=[fairness_constrains], 
               methods=[method], 
               file="/work/ws/nemo/fr_tf167-conda-0/autosklearn", 
               seeds= [seed], 
               sf=[sf],
               runtime = 2000000, 
              runcount=runcount, 
              under_folder=under_folder, 
              performance = performance,
              test=True)
