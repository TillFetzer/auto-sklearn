from fanova import fANOVA
import fanova.visualizer
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import json
import os
import numpy as np
import pandas as pd
import pickle







def format_data(method, constrain, dataset,  data_path, y_format):
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
            X.append(list(ds["configs"][str(i+1)].values()))
            
            #print(list(ds["configs"][str(i+1)].values())[:-1])
            y_index = 0 if y_format == "performance" else 1
            Y.append(d[1][0][y_index])
    X = pd.DataFrame(X)
    #that could be more difficult for different methods
    
    return X, pd.DataFrame(Y)
           
        

if __name__ == '__main__':
    method = "moo_ps_ranker"
    constrain = "demographic_parity"
    data_path = "/home/till/Desktop/cross_val/"
    dataset = "adult"
    y_format = "performance"
    X,Y = format_data(method, constrain, dataset, data_path, y_format) 
    # create an instance of fanova with data for the random forest and the configSpace
    file = open("/home/till/Documents/auto-sklearn/tmp/moo_ps_config_space.pickle",'rb')
    cs = pickle.load(file)
    X.columns = cs.get_hyperparameter_names()
    f = fANOVA(X = X, Y = Y, config_space = cs)

    # marginal for first parameter
    p_list = (0, )
    res = f.quantify_importance(p_list)
    print(res)
    # directory in which you can find all plots
    #plot_dir =  './tmp/'
    # first create an instance of the visualizer with fanova object and configspace
    #vis = fanova.visualizer.Visualizer(f, cs, plot_dir)
    # generating plot data for col0
    #mean, std, grid = vis.generate_marginal(0)

    # creating all plots in the directory
    #vis.create_all_plots()

