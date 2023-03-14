from run_experiment import run_experiment
import argparse, sys
parser=argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
args=parser.parse_args()

idx = args.idx
seeds = [12345,25,42,45451, 97]
methods = ["lfr"]
datasets = ["adult","compass", "german", "lawschool"]
sfs = ["sex", "race", "foreign_worker","race"]
fairness_constrains=["demographic_parity","equalized_odds"]
dataset = datasets[int(idx/15)]
sf = sfs[int(idx/15)]
method = methods[int(idx/5)%3]
seed = seeds[idx%5]


#rint(dataset)
#print(method)
#print(seed)
run_experiment(datasets =[dataset], fairness_constrains=["equalized_odds"], methods=[method], file="/work/dlclarge2/fetzert-MySpace/autosklearn", seeds= [seed], sf=[sf] ,runtime = 3600, runcount=200)
