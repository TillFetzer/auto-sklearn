from fanova import fANOVA
import fanova.visualizer
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
import json
import os
import numpy as np
import pandas as pd
import pickle
import argparse






def format_data(method, constrain, dataset,  data_path, y_format, ocs):
    """
    in: method, constraint, data_path
    do: read data from data_path and format it for fanova
    out: fromated X,Y,cs
    """
    X = []
    Y = []
    base_file = data_path + constrain + "/" + dataset
    for seed in os.listdir(base_file):
        file  = "{}/{}/{}/200timesstrat/runhistory.json".format(base_file, seed, method)         
        with open(file) as f:
            ds = json.load(f)
        for i, d in enumerate(ds["data"]):
            config = []
            for i,x in enumerate(list(ds["configs"][str(i+1)].values())[0:13]):
                keys = list(ds["configs"][str(i+1)])
                try:
                    config.append(float(x))
                except:
                    if isinstance(ocs[keys[i]], CategoricalHyperparameter):
                        choices = list(getattr(ocs[keys[i]], "choices"))
                        config.append(choices.index(x) if x in choices else ValueError("not in choices")) 
                    elif isinstance(ocs[keys[i]], Constant):
                        #Constant Hyperparameter have no fanova value
                        config.append(0)
                    else:
                        raise ValueError("Funny Hyperparameter")
            
            X.append(config)
            #print(list(ds["configs"][str(i+1)].values())[:-1])
            y_index = 0 if y_format == "performance" else 1
            if constrain == "error_rate_difference":
                Y.append(d[1][0][y_index])
            else:
                Y.append(1-d[1][0][y_index])
    X = pd.DataFrame(X)
    #that could be more difficult for different methods
    X.columns = list(ds["configs"][str(i+1)].keys())[0:13]

    return X, pd.DataFrame(Y)
           
        

if __name__ == '__main__':
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    parser.add_argument("--uf", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    args=parser.parse_args()
    idx = args.idx
    under_folder = args.uf
    data_path = args.data_path
    save_path = args.save_path
    methods = ["moo+cr","moo+ps","moo+ps+cr","ps+cr+lfr"]
    #methods = ["moo","so","ps","cr", "moo+cr", "moo_ps","moo+ps+cr","moo+ps*cr","lfr"]
    datasets = ["german","adult","lawschool","compass"]
    fairness_constrains=["consistency_score","demographic_parity","equalized_odds", "error_rate_difference"]
    #performance = utils_fairlearn.set_f1_score()
    y_formats = ["performance","fairness"]
    files = ["moo_cr","moo_ps","moo_ps_cr","moo_ps_cr_lfr"]
    method = methods[int(idx)%len(methods)]
    file = files[int(idx)%len(methods)]
    y_format = y_formats[int(idx/(len(methods)))%len(y_formats)]
    dataset = datasets[int(idx/(len(methods)*len(y_formats)))%len(datasets)]
    constrain = fairness_constrains[int(idx/(len(methods)*len(y_formats)*len(datasets)))%len(fairness_constrains)]
    print(method, file, y_format, dataset, constrain)
    cs_file = open("{}/{}_config_space.pickle".format(under_folder,file),'rb')
    ocs = pickle.load(cs_file)
    
    X,Y = format_data(method, constrain, dataset, data_path, y_format,ocs) 
           
           
    cs = ConfigurationSpace()
    for hp in X.columns:
        cs.add_hyperparameter(ocs[hp])   
                #if categorical change to hash
    assert len(cs.get_hyperparameters()) == len(X.columns)
    f = fANOVA(X = X, Y = np.array(Y), config_space = cs)
    print("{}, {}, {}".format(dataset, constrain, method))
    best_p_margs = f.get_most_important_pairwise_marginals(n=10)
    with open("{}/{}_{}_results.txt".format(save_path, file, y_format), "w") as f:
        f.write("dataset: {}, constrain: {}, method: {}\n".format(dataset, constrain, method))       
        f.write("best pairwise marginals: {}\n".format(best_p_margs))

    """
    for dataset in ["lawschool", "adult", "compass","german"]:
        for constrain in ["equalized_odds", "demographic_parity", "consistency_score", "error_rate_difference"]:
            method = "moo+ps+cr+lfr"
            data_path = "/home/till/Desktop/cross_val/"
            y_format = "fairness"
            file = open("/home/till/Documents/auto-sklearn/tmp/moo_ps_cr_lfr_config_space.pickle",'rb')
            ocs = pickle.load(file)
            X,Y = format_data(method, constrain, dataset, data_path, y_format,ocs) 
            #X = X.drop(columns = ["balancing:strategy"])
            # create an instance of fanova with data for the random forest and the configSpace
            cs = ConfigurationSpace()
            for hp in X.columns:
                #if  isinstance(ocs[hp], CategoricalHyperparameter):
                #    setattr(ocs[hp], "choices", tuple(abs(hash(x)%10) for x in getattr(ocs[hp],"choices")))
                #    setattr(ocs[hp], "default_value", abs(hash(getattr(ocs[hp],"choices")[0])%10))
                cs.add_hyperparameter(ocs[hp])   
                #if categorical change to hash
            assert len(cs.get_hyperparameters()) == len(X.columns)
            f = fANOVA(X = X, Y = np.array(Y), config_space = cs)
            print("{}, {}, {}".format(dataset, constrain, method))
            # marginal for first parameter
            #for i in list(cs.get_hyperparameters()):
                #if i.name != "fair_preprocessor:__choice__":
                #    continue
                #res = f.quantify_importance([i.name])
                #print(res)
            res = f.quantify_importance(['fair_preprocessor:__choice__', 'classifier:random_forest:max_features'])
            print(res)
            #p_list = (0,1,2,3,4,5,6,7,8,9,10,11,12,)
            #res = f.quantify_importance(p_list)
            #print(res)

            #best_p_margs = f.get_most_important_pairwise_marginals(n=5)
            

            # directory in which you can find all plots
            #plot_dir =  '/home/till/Documents/auto-sklearn/tmp/plots/moo+ps+cr+lfr/{}/{}'.format(constrain, dataset)
            # first create an instance of the visualizer with fanova object and configspace
            #label = "1-{}".format(constrain) if constrain != "error_rate_difference" else "{}".format(constrain)
            #vis = fanova.visualizer.Visualizer(f, cs, plot_dir, y_label="{}".format(constrain))
            # generating plot data for col0
            #vis.generate_marginal(3)

            # creating all plots in the directory
            #vis.create_all_plots()

