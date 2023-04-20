import json
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
from collections import defaultdict
from autosklearn.util.multiobjective import pareto_front
import numpy as np 
import copy
import corrleation_remover
import argparse, sys
from autosklearn.pipeline.components.classification.random_forest import RandomForest
from ConfigSpace.configuration_space import Configuration
import copy
def pareto_set(all_costs):
    confs = all_costs["configs"]
    all_costs = np.array(all_costs["points"])
    sort_by_first_metric = np.argsort(all_costs[:, 0])
    efficient_points = pareto_front(all_costs, is_loss=True)
    pareto_set = []
    pareto_config =[]
    for argsort_idx in sort_by_first_metric:
            if not efficient_points[argsort_idx]:
                continue
            pareto_set.append(all_costs[argsort_idx,:])
            pareto_config.append(confs[argsort_idx])
    pareto_set = pd.DataFrame(pareto_set)
    #if len(pareto_set.index)<1:
    #    pareto_set.loc[-1] = [-1, pareto_set[1][0]]
    #    pareto_set.index = pareto_set.index + 1  # shifting index
    #    pareto_set = pareto_set.sort_index()
    #    pareto_set =  pareto_set.append([[2, pareto_set[1][0]]])
    return pareto_set, pareto_config

def load_config(file):
    data = defaultdict()  
    data["points"] = []
    data["configs"] = []
    with open(file) as f:
        ds = json.load(f)
        for d in ds["data"]:
            try:
                if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                    continue
                point = d[1][0]
                config = ds["configs"][str(d[0][0])]
                    

                #if run was not sucessfull no train loss is generated
                #these happened also for sucessfull runs for example timeout
            except KeyError:
                continue 
            data['points'].append(point)
            data['configs'].append(config)
    data['points'] = pd.DataFrame(data['points'])
    return data["points"], data["configs"]


def create_cr_reruns(
source_folder= "/home/till/Documents/auto-sklearn/tmp", 
dataset="adult", 
constrain="consistency_score", 
seed=13, 
method="moo", 
runetime="200timesstrat",
goal_folder = "base",
rf_seed = 1
):
    data = defaultdict()   
    file  = "{}/{}/{}/{}/{}/{}/{}/runhistory.json".format(source_folder,goal_folder,constrain,dataset,seed,method,runetime)
    data["points"], data["configs"]  = load_config(file)
    data['pareto_set'], data['pareto_config']  = pareto_set(data)
    RandomForest.activate_random_seed()
    cr_configs = []
    for config in data["pareto_config"]:
        conf = copy.deepcopy(config)
        conf['feature_preprocessor:__choice__'] = "CorrelationRemover"           
        for i in range(1,11):
            conf1 = copy.deepcopy(conf)
            conf1['feature_preprocessor:CorrelationRemover:alpha'] = i*0.1
            #these righ now does not help            
            conf1['classifier:random_forest:random_state_forest'] = rf_seed    
            cr_configs.append(conf1)
                
    #num_configs = len(data['pareto_config'])*10 + 1
    tmp =  "{}/{}/{}/{}/{}/cr/{}/{}/del/".format(source_folder,goal_folder,constrain,dataset,seed,"one_rf_seed", rf_seed)
    # these is run was not finished
    if os.path.exists(tmp):
        data_2 = defaultdict() 
        if len(os.listdir(tmp)) == 0:
            return
        data_2["points"], data_2["configs"] = load_config(tmp + "smac3-output/run_{}/runhistory.json".format(seed))
        cr_configs = cr_configs[(len(data_2['configs'])-1):]

    num_configs = len(cr_configs) + 1 
    corrleation_remover.run_experiment(
        dataset,
        constrain, 
        sf,  
        20000000, #these is only that it does stop 
        source_folder,
        seed, 
        num_configs, #these stops it
        goal_folder,
        configs = cr_configs,
        rf_seed = rf_seed)
                  



   



if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    parser.add_argument("--sf", type=str)
    parser.add_argument("--gf", type=str)
    args=parser.parse_args()
    idx = args.idx
    source_folder = args.sf
    goal_folder = args.gf
    seeds = [12345,25,42,45451,97,13,27,39,41,53]
    methods = ["moo"]
    datasets = ["german","adult","compass","lawschool"]
    #datasets = ["german"]
    sfs = ["personal_status", "sex", "race", "race"]
    fairness_constrains=["demographic_parity","equalized_odds", "consistency_score", "error_rate_difference"]
    rf_seeds = [1,2,3,4,5]
    dataset = datasets[int(idx/(len(seeds)*len(methods)*len(rf_seeds)))%len(datasets)]
    sf = sfs[int(idx/(len(seeds)*len(methods)*len(rf_seeds)))%len(datasets)]
    method = methods[int(idx/(len(seeds)*len(rf_seeds)))%len(methods)]
    seed = seeds[int(idx/len(rf_seeds))%len(seeds)]
    fairness_constrains = fairness_constrains[int(idx/(len(seeds)*len(methods)*len(datasets)*len(rf_seeds)))]
    rf_seed = rf_seeds[idx%len(rf_seeds)]
    print(fairness_constrains)
    print(dataset)
    print(seed)    
    print(rf_seed)
    data = create_cr_reruns(
        source_folder= source_folder, 
        dataset=dataset, 
        constrain=fairness_constrains, 
       seed=seed, 
        method=method, 
        runetime="200timesstrat",
       goal_folder = goal_folder,
       rf_seed = rf_seed
      )
 