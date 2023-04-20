from run_experiment import run_experiment
import argparse, sys
parser=argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
parser.add_argument("--uf", type=str)
args=parser.parse_args()

idx = args.idx
under_folder = args.uf
seeds = [12345,25,42,45451, 97,13,27,39,41,53]
methods = ["moo"]
datasets = ["german","adult","compass","lawschool"]
#datasets = ["german"]
sfs = ["personal_status", "sex", "race", "race"]
fairness_constrains=["equal_opportunity_difference"]

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
run_experiment(datasets =[dataset], fairness_constrains=[fairness_constrains], methods=[method], file="/work/dlclarge2/fetzert-MySpace/autosklearn", seeds= [seed], sf=[sf] ,runtime = 200000, runcount=runcount, under_folder=under_folder)
