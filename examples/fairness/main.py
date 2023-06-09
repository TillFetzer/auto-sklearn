from run_experiment import run_experiment
import argparse, sys
import autosklearn
parser=argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
parser.add_argument("--uf", type=str)
args=parser.parse_args()

idx = args.idx
under_folder = args.uf
seeds = [12345,25,42,45451, 97,13,27,39,41,53]
#methods = ["lfr","moo+lfr","moo+ps+cr+lfr"]
methods = ["moo","so","ps","cr", "moo+cr", "moo_ps","moo+ps+cr","moo+ps*cr","lfr"]
datasets = ["german","adult","lawschool","compass"]
#datasets = ["german"]
sfs = ["personal_status", "sex", "race", "race"]
fairness_constrains=["consistency_score","demographic_parity","equalized_odds", "error_rate_difference"]
performance = autosklearn.metrics.f1_macro

dataset = datasets[int(idx/(len(seeds)*len(methods)))%len(datasets)]
sf = sfs[int(idx/(len(seeds)*len(methods)))%len(datasets)]
method = methods[int(idx/len(seeds))%len(methods)]
seed = seeds[idx%len(seeds)]
fairness_constrains = fairness_constrains[int(idx/(len(seeds)*len(methods)*len(datasets)))]

runcount = 200


print(dataset)
print(method)
print(seed)
print(fairness_constrains)
run_experiment(datasets =[dataset], fairness_constrains=[fairness_constrains], methods=[method], file="/work/dlclarge2/fetzert-MySpace/autosklearn", seeds= [seed], sf=[sf] ,runtime = 200000, runcount=runcount, under_folder=under_folder, performance = performance)
