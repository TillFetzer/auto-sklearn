import base
import corrleation_remover
import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--d", help="datsets",nargs="*")
parser.add_argument("--m", help="Foo the program",nargs="*")
parser.add_argument("--r", help="Foo the program", type = int)
parser.add_argument("--fc", help="Foo the program", nargs="*")
parser.add_argument("--sa", help="Foo the program", nargs="*")
parser.add_argument("--f")
args=parser.parse_args()

datasets = args.d
fairness_constrains = args.fc
runtime = args.r
methods = args.m
file = args.f
sf = args.sa # same length then dataset or 1
# sf= ["foreign_worker"]
print("start")
if len(sf) == 1:
    sf = len(datasets) * sf
for constrain in fairness_constrains:
    for i, dataset in enumerate(datasets):
        for method in methods:
            if method == "moo":
                base.run_experiment(dataset, constrain, sf[i], runtime, file)
            if method == "cr":
                corrleation_remover.run_experiment(dataset, constrain, sf[i], runtime, file)
        print("all runs of {} finished".format(dataset))
print("finished")
