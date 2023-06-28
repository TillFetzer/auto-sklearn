from fanova import fANOVA
import fanova.visualizer
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
import json
import os
import numpy as np
import pandas as pd
import pickle







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
            for i,x in enumerate(list(ds["configs"][str(i+1)].values())[0:4]):
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
            Y.append(1-d[1][0][y_index])
    X = pd.DataFrame(X)
    #that could be more difficult for different methods
    X.columns = list(ds["configs"][str(i+1)].keys())[0:4]

    return X, pd.DataFrame(Y)
           
        

if __name__ == '__main__':
    for dataset in ["lawschool", "adult", "compass","german"]:
        for constrain in ["equalized_odds", "demographic_parity", "consistency_score", "error_rate_difference"]:
            method = "moo+ps+cr+lfr"
            data_path = "/home/till/Desktop/cross_val/"
            y_format = "fairness"
            file = open("/home/till/Documents/auto-sklearn/tmp/moo_ps_cr_lfr_config_space.pickle",'rb')
            ocs = pickle.load(file)
            X,Y = format_data(method, constrain, dataset, data_path, y_format,ocs) 
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

            # marginal for first parameter
            p_list = ("fair_preprocessor:__choice__",)
            res = f.quantify_importance(p_list)
            print(res)

            #best_p_margs = f.get_most_important_pairwise_marginals(n=3)
            #print(best_p_margs)

            # directory in which you can find all plots
            plot_dir =  '/home/till/Documents/auto-sklearn/tmp/plots/moo+ps+cr+lfr/{}/{}'.format(constrain, dataset)
            # first create an instance of the visualizer with fanova object and configspace
            label = "1-{}".format(constrain) if constrain != "error_rate_difference" else "{}".format(constrain)
            vis = fanova.visualizer.Visualizer(f, cs, plot_dir, y_label="{}".format(constrain))
            # generating plot data for col0
            #vis.generate_marginal(3)

            # creating all plots in the directory
            vis.create_all_plots()

