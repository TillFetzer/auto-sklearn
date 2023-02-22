from run_experiment import run_experiment
import argparse, sys
parser=argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
args=parser.parse_args()

idx = args.idx
seeds = [42,12345,25,97,45451]
methods = ["moo", "redlineing", "cr"]
datasets = ["adult","compass", "german"]
dataset = datasets[int(idx/15)]
method = methods[int(idx/5)%3]
seed = seeds[idx%5]
#rint(dataset)
#print(method)
#print(seed)
run_experiment(datasets =[dataset], fairness_constrains=["demographic_parity"], methods=[method], file="/work/dlclarge2/fetzert-MySpace/autosklearn", seeds= seed, sf=["sex"] ,runtime = 10800)
