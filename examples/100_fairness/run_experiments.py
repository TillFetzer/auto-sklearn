import base
import corrleation_remover

datasets = ["lawschool"]
fairness_constrains = ["demographic_parity"]
runtime = 3 * 60 * 60
methods = ["moo", "cr"]
sf = ["sex"]  # same length then dataset or 1
# sf= ["foreign_worker"]
print("start")
if len(sf) == 1:
    sf = len(datasets) * sf
for constrain in fairness_constrains:
    for i, dataset in enumerate(datasets):
        for method in methods:
            if method == "moo":
                base.run_experiment(dataset, constrain, sf[i], runtime)
            if method == "cr":
                corrleation_remover.run_experiment(dataset, constrain, sf[i], runtime)
        print("all runs of {} finished".format(dataset))
print("finished")
