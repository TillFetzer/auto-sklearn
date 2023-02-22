import base
import corrleation_remover
import redlineing
def run_experiment(datasets =["adult"], fairness_constrains=["demographic_parity"], methods=["moo"], file="/work/dlclarge2/fetzert-MySpace/autosklearn", seeds=[42], sf=["sex"] ,runtime = 10800):
    # sf= ["foreign_worker"]
    print("start")
    if len(sf) == 1:
        sf = len(datasets) * sf
    for constrain in fairness_constrains:
        for i, dataset in enumerate(datasets):
            for seed in seeds:
                for method in methods:
                    if method == "moo":
                        base.run_experiment(dataset, constrain, sf[i], runtime, file, seed)
                    if method == "redlineing":
                        redlineing.run_experiment(dataset, constrain, sf[i], runtime, file, seed)
                    if method == "cr":
                        corrleation_remover.run_experiment(dataset, constrain, sf[i], runtime, file, seed)
            print("all runs of {} finished".format(dataset))
    print("finished")
