import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from statistics import mean
from eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot


def plot(data,ax,**kwargs,):
        if ax is None:
            ax = plt.gca()
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        x, y = (data[0], data[1])
        ax.scatter(x, y, **kwargs)
        return ax 
def pareto_plot(data,ax,**kwargs,):
      
    ax =  plot(data,ax=ax, **kwargs) 
    if ax is None:
                ax = plt.gca()

    if "marker" not in kwargs:
        kwargs["marker"] = "o"
    xs,ys = (data[0], data[1])
    new_xs = list(xs)
    new_ys = list(ys)
     # line plots don't allow these
    for k in ["facecolor", "edgecolor"]:
        if k in kwargs:
            del kwargs[k]
    
    for k, k_new in [("s", "markersize")]:
        if k in kwargs:
            kwargs[k_new] = kwargs.pop(k)
    ax.step(new_xs, new_ys, where="post", **kwargs)

    return ax

def make_plot(methods=["moo","ps"], dataset= "adult", runtime=10800):
    sns.set_context("paper", font_scale=0.6)
    figsize = (20, 8)
    dpi = 300
    main_size = 20
    plot_offset = 0.05
    title_size = 18
    label_size = 16
    tick_size = 12
    with open('/home/till/Documents/auto-sklearn/results_t.jmoo_psn') as f:
        data = jmoo_psn.load(f)
    data = pd.DataFrame(data)
    result_0 = data.query("dataset == @dataset and methods == @methods[0] and runtime==@runtime")
    result_1 = data.query("dataset == @dataset and methods == @methods[1] and runtime==@runtime")


    result_0_val = pd.DataFrame(result_0["results"][result_0.index[0]]["val"])
    result_0_test = pd.DataFrame(result_0["results"][result_0.index[0]]["test"])
    result_1_val = pd.DataFrame(result_1["results"][result_1.index[0]]["val"])
    result_1_test = pd.DataFrame(result_1["results"][result_1.index[0]]["test"])
    
    fig, (val_ax, test_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
    )
    fig.supxlabel(result_0["performance_metrics"][result_0.index[0]],fontsize=label_size)
    fig.supylabel(result_0["fairness_metrics"][result_0.index[0]].replace("_", " ").capitalize(), fontsize=label_size)

    val_ax.set_title("Validation", fontsize=title_size)
    test_ax.set_title("Test", fontsize=title_size)
    for ax in (val_ax, test_ax):
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.set_box_aspect(1)

    def rgb(r: int, g: int, b: int) -> str:
        return "#%02x%02x%02x" % (r, g, b)


    c_chocolate = rgb(217, 95, 2)

    alpha = 1

    styles = {
        "moo_points": dict(s=15, marker="o", edgecolors=c_chocolate, facecolors="none"),
        "moo_pareto": dict(s=4, marker="o", color=c_chocolate, linestyle="dotted", linewidth=2),
        "ps_points": dict(s=15, marker="o", edgecolors="black", facecolors="none"),
        "ps_pareto": dict(s=4, marker="o", color="black", linestyle="-", linewidth=2, line ="dotted"),
        "ps_test_points": dict(s=15, marker="o", edgecolors="black", facecolors="none", color ="black"),
        "moo_test_points": dict(s=15, marker="o", edgecolors=c_chocolate,color=c_chocolate, facecolors="none"),
    }


    ax = val_ax

    # Highlight val pareto front and how things moved
    
    plot(result_0_val, ax=ax, **styles["moo_points"])
    pareto_plot(result_0_val, ax=ax, **styles["moo_pareto"])
    # Show the test pareto but faded
    plot(result_1_val, ax=ax, **styles["ps_points"])
    pareto_plot(result_1_val, ax=ax, **styles["ps_pareto"])
    # test_scores.plot( ax=ax, alpha=alpha, **styles["test_points"])

    ax = test_ax

    # Highlight val pareto front and how things moved
    
    plot(result_0_test, ax=ax, **styles["moo_test_points"])
   
    # Show the test pareto but faded
    plot(result_1_test, ax=ax, **styles["ps_test_points"])
   
    # test_scores.plot( ax=ax, alpha=alpha, **styles["test_points"])

    min_x = min(min(result_0_val[0]), min(result_0_test[0]), min(result_1_val[0]), min(result_1_test[0]))
    min_y = min(min(result_0_val[1]), min(result_0_test[1]), min(result_1_val[1]), min(result_1_test[1]))
    max_x = max(max(result_0_val[0]), max(result_0_test[0]), max(result_1_val[0]), max(result_1_test[0]))
    max_y = max(max(result_0_val[1]), max(result_0_test[1]), max(result_1_val[1]), max(result_1_test[1]))

    dx = abs(max_x - min_x)
    dy = abs(max_y - min_y)

    for ax in (val_ax, test_ax):
        ax.set_xlim(min_x - dx * plot_offset, max_x + dx * plot_offset)
        ax.set_ylim(min_y - dy * plot_offset, max_y + dy * plot_offset)
    # We're adding the legend in tex code
    # ax.legend()
    fig.suptitle(dataset, fontsize=main_size)
    legend_elements = [Line2D([0], [0], color=c_chocolate, lw=4, label='moo without preprocessing'),
                   Line2D([0], [0], color="black", lw=4, label='moo with correlation remover')]
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.legend(handles=legend_elements, loc=3,  prop={'size': 6})
    plt.show()
    plt.savefig(f"./figures/experiment_1_{dataset}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)



def pareto_set(all_costs):
    confs = all_costs["configs"]
    all_costs = np.array(all_costs["points"])
    moo_psrt_by_first_metric = np.argsort(all_costs[:, 0])
    efficient_points = pareto_front(all_costs, is_loss=True)
    pareto_set = []
    pareto_config =[]
    for argmoo_psrt_idx in moo_psrt_by_first_metric:
            if not efficient_points[argmoo_psrt_idx]:
                continue
            pareto_set.append(all_costs[argmoo_psrt_idx,:])
            pareto_config.append(confs[argmoo_psrt_idx])
    pareto_set = pd.DataFrame(pareto_set)
    return pareto_set, pareto_config




def load_data(filepath, runetime, methods):
    data = defaultdict()
    for constrain in os.listdir(filepath):
        data[constrain] = defaultdict()
        constrain_path = "{}{}".format(filepath, constrain)
        for dataset in os.listdir(constrain_path):
            data[constrain][dataset] = defaultdict()
            dataset_path = "{}/{}".format(constrain_path, dataset)
            for seed in os.listdir(dataset_path):
                data[constrain][dataset][seed] = defaultdict()
                seed_path = "{}/{}".format(dataset_path, seed)
                for method in methods:
                    data[constrain][dataset][seed][method] = defaultdict()
                    data[constrain][dataset][seed][method]["points"] = []
                    data[constrain][dataset][seed][method]["configs"] = []
                   
                    runetime_folder = "same_hyperparamer"  if method == "ps_transfere" else runetime
                    
                    
                    file  = "{}/{}/{}/runhistory.json".format(seed_path,method,runetime_folder)
                    if not(os.path.exists(file)):
                                    file  = "{}/{}/{}/del/smac3-output/run_{}/runhistory.json".format(seed_path,method,runetime_folder,seed) 
                    with open(file) as f:
                        ds = json.load(f)
                    for d in ds["data"]:
                        try:
                            if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                                continue
                           
                            point = d[1][0]
                            config = ds["configs"][str(d[0][0])]
                            if method == "so":
                                p = []
                                p.append(point)
                                ff  = "{}/{}/{}/fairness.json".format(seed_path,method,runetime_folder)
                                with open(ff) as f:
                                    fairness = json.load(f)
                                if constrain == "consistency_score":
                                    p.append(1-fairness[d[0][0] -1])
                                else:
                                    p.append(fairness[d[0][0] -1])
                                point = p
                                
                        #if run was not sucessfull no train loss is generated
                        #these happened almoo_ps for sucessfull runs for example timeout
                        except KeyError:
                            continue 
                        
                        data[constrain][dataset][seed][method]['points'].append(point)
                        data[constrain][dataset][seed][method]['configs'].append(config)
                    data[constrain][dataset][seed][method]['points'] = pd.DataFrame(data[constrain][dataset][seed][method]['points'])
                    data[constrain][dataset][seed][method]['pareto_set'], data[constrain][dataset][seed][method]['pareto_config']   = pareto_set(data[constrain][dataset][seed][method])

    return data


def  load_data_particully(filepath , 
    runetime, 
    datasets = ["german"], 
    constrains = ["demographic_parity"], 
    seeds= [42], 
    folders = ["same_hyperparameter","hopefully_last_ruhopefully_last_run", "white_line"]):
    data = defaultdict()
    for constrain in constrains:
        data[constrain] = defaultdict()
        constrain_path = "{}{}".format(filepath, constrain)
        for dataset in datasets:
            data[constrain][dataset] = defaultdict()
            dataset_path = "{}/{}".format(constrain_path, dataset)
            for seed in seeds:
                data[constrain][dataset][seed] = defaultdict()
                seed_path = "{}/{}".format(dataset_path, seed)
                methods = ["moo", "ps"]
                for method in methods:
                    points = []
                    configs = []
                    
                    runetime_folders = folders  if method == "ps" else [runetime]
                    for runetime_folder in runetime_folders:
                        method_path = "{}/{}/{}".format(seed_path, method, runetime_folder)
                        if not(os.path.exists(method_path)):
                            continue
                        set = False
                        length = 0
                        for rf_seed in os.listdir(method_path):
                           
                            
                            #print(method)
                            #method = 'ps'
                            data[constrain][dataset][seed][method] = defaultdict()
                            data[constrain][dataset][seed][method]["points"] = []
                            data[constrain][dataset][seed][method]["configs"] = []
                            if rf_seed == "runhistory.jmoo_psn" or rf_seed == "del":
                                file  = "{}/{}/{}/runhistory.jmoo_psn".format(seed_path,method,runetime_folder) 
                            else:
                                file  = "{}/{}/{}/{}/runhistory.jmoo_psn".format(seed_path,method,runetime_folder, rf_seed) 
                                if not(os.path.exists(file)):
                                    file  = "{}/{}/{}/{}/del/smac3-output/run_{}/runhistory.jmoo_psn".format(seed_path,method,runetime_folder, rf_seed, seed) 
                            try:
                                with open(file) as f:
                                    ds = jmoo_psn.load(f)
                            except: 
                                print(file + "not exists")
                            ps = []
                            cs = []
                            for d in ds["data"]:
                                try:
                                    if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                                        continue
                                    if method == "ps" and ds['config_origins'][str(d[0][0])] != "Initial design":
                                        continue
                                    point = d[1][0]
                                    config = ds["configs"][str(d[0][0])]

                                #if run was not sucessfull no train loss is generated
                                #these happened almoo_ps for sucessfull runs for example timeout
                                except KeyError:
                                    continue 
                                ps.append(point)
                                cs.append(config)
                            if len(ps) > length and not(set):
                                length = len(ps)
                                set = True
                            if len(ps) != length:
                                print(file)
                                continue
                            points.append(ps)
                            configs.append(cs)
                           

                            
                    points = np.array(points)
                    data[constrain][dataset][seed][method]['points']= pd.DataFrame(np.mean(points, axis = 0)) 
                    #data[constrain][dataset][seed][method]['points'] = pd.DataFrame(data[constrain][dataset][seed][method]['points'])
                    data[constrain][dataset][seed][method]['configs'] = configs[0]
                    if method == "moo":
                        data[constrain][dataset][seed][method]['pareto_set'], data[constrain][dataset][seed][method]['pareto_config']   = pareto_set(data[constrain][dataset][seed][method])
                    #print("file:{},pareto_set:{}".format(file, data[constrain][dataset][seed][method]['points']))
    return data



def make_plot_2(data):
    #TODO add last and first point
    sns.set_context("paper", font_scale=0.6)

    #TODO set on big monitor
    figsize = (27,8)
    dpi = 300
    main_size = 20
    plot_offset = 0.1
    title_size = 18
    label_size = 16
    tick_size = 12
   
   
    
    fig, axis = plt.subplots(
        len(data),
        len(data[list(data.keys())[0]]),
        1, #needs to be more flexible for nowe is ok
        sharey=True,
        figsize=figsize,
    )
    fig.supxlabel("error",fontsize=label_size)
    fig.supylabel("error_rate_difference", fontsize=label_size)
    
    alpha = 0.1

    styles = {
        "moo_points": dict(s=15, marker="o", color="red"),
        "moo_pareto": dict(s=4, marker="o", color="red", linestyle="-", linewidth=2),
        "ps_points": dict(s=15, marker="o", color="blue"),
        "ps_pareto": dict(s=4, marker="o", color="blue", linestyle="-", linewidth=2),
        "redlineing_points": dict(s=15, marker="o", color ="green"),
        "redlineing_pareto": dict(s=4, marker="o", color="green", linestyle="-", linewidth=2)
    }
    for i,constrain in enumerate(data.keys()):
        global_max_y = 0
        global_min_y = 1
        for j,dataset in enumerate(data[constrain].keys()):
            global_min_x = 1
            global_max_x = 0
            
            if len(data.keys()) == 1:
                ax = axis[j]
            else:
                ax = axis[i,j]
            
            ax.set_title(dataset, fontsize=title_size)
            for seed in data[constrain][dataset].keys():
                #seed = "25" 
                moo_points = data[constrain][dataset][seed]['moo']['points']
                ps_points = data[constrain][dataset][seed]['ps']['points']
                redlineing_points = data[constrain][dataset][seed]['redlineing']['points']
                moo_pf = data[constrain][dataset][seed]['moo']['points']
                ps_pf = data[constrain][dataset][seed]['ps']['points']
                redlineing_pf = data[constrain][dataset][seed]['redlineing']['points']
                plot(moo_points, ax=ax, **styles["moo_points"], alpha = alpha)
                #eaf_plot = EmpiricalAttainmentFuncPlot()
                pareto_plot(moo_pf, ax=ax, **styles["moo_pareto"])
                plot(ps_points, ax=ax, **styles["ps_points"], alpha = alpha)
                pareto_plot(ps_pf, ax=ax, **styles["ps_pareto"])
                plot(redlineing_points, ax=ax, **styles["redlineing_points"], alpha = alpha)
                pareto_plot(redlineing_pf, ax=ax, **styles["redlineing_pareto"])
                if len(moo_pf.index)>3:
                    moo_pf.drop(index=moo_pf.index[[0,-1]],inplace=True)
                if len(ps_pf.index)>3:
                    ps_pf.drop(index=ps_pf.index[[0,-1]],inplace=True)
                if len(redlineing_pf.index)>3:
                    redlineing_pf.drop(index=redlineing_pf.index[[0,-1]],inplace=True)
                local_min_x = min(min(moo_pf[0]), min(ps_pf[0]), min(redlineing_pf[0]))
                local_min_y = min(min(moo_pf[1]), min(ps_pf[1]), min(redlineing_pf[1]))
                local_max_x = max(max(moo_pf[0]), max(ps_pf[0]), max(redlineing_pf[0]))
                local_max_y = max(max(moo_pf[1]), max(ps_pf[1]), max(redlineing_pf[1]))
                #local_min_x = min(min(moo_points[0]), min(ps_points[0]), min(redlineing_points[0]))
                #local_min_y = min(min(moo_points[1]), min(ps_points[1]), min(redlineing_points[1]))
                #local_max_x = max(max(moo_points[0]), max(ps_points[0]), max(redlineing_points[0]))
                #local_max_y = max(max(moo_points[1]), max(ps_points[1]), max(redlineing_points[1]))
                global_min_y = local_min_y if local_min_y < global_min_y  else global_min_y
                global_min_x = local_min_x if local_min_x < global_min_x  else global_min_x
                global_max_y = local_max_y if local_max_y > global_max_y  else global_max_y
                global_max_x = local_max_x if local_max_x > global_max_x  else global_max_x
            dx = abs(global_max_x - global_min_x) if abs(global_max_x - global_min_x) > 0 else 0.01
            ax.set_xlim(max(global_min_x - dx * plot_offset,0), global_max_x + dx * plot_offset)
            dy = abs(global_max_y - global_min_y)
            ax.set_ylim(max(global_min_y - dy*plot_offset,0), global_max_y +  dy * plot_offset)
            ax.tick_params(axis="both", which="major", labelsize=tick_size)
        for j,dataset in enumerate(data[constrain].keys()):
            if len(data.keys()) == 1:
                ax = axis[j]
            else:
                ax = axis[i,j]
           
            #ax.set_box_aspect(1)
    
    legend_elements = [Line2D([0], [0], color="red", lw=4, label='moo without preprocessing'),
                   Line2D([0], [0], color="blue", lw=4, label='moo with correlation remover'),
                    Line2D([0], [0], color="green", lw=4, label='moo without SA and corrleation remover')
                    ]
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.98])
    fig.legend(handles=legend_elements, loc=3,  prop={'size': 8})


  
    plt.show()
    #plt.savefig(f"./figures/experiment_1_{dataset}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)

def plots_we(pf, ax, color):
    #
    levels = [len(pf[0]) // 4, len(pf[0]) // 2, 3* len(pf[0]) //4]
    surfs_list = [get_empirical_attainment_surface(costs=costs, levels=levels) for costs in pf]
    #_,we = plt.Subplot(fig, ax)
    eaf_plot = EmpiricalAttainmentFuncPlot()
    eaf_plot.plot_multiple_surface_with_band(
                ax,
                surfs_list=surfs_list,
                colors=color,
                labels= [ x for x in range(len(surfs_list))]
                )
   

    
def make_plot_3(data):
    sns.set_context("paper", font_scale=0.6)
    #load_data()
    figsize = (27,10)
    dpi = 300
    main_size = 20
    plot_offset = 0.1
    title_size = 18
    label_size = 16
    tick_size = 12
   
   
    
    fig, axis = plt.subplots(
        nrows=len(data),
        ncols=len(data[list(data.keys())[0]]),  #needs to be more flexible for nowe is ok
        #sharey=True,
        figsize=figsize,
    )
    
    fig.supxlabel("error",fontsize=label_size)
    fig.supylabel("1_consistency_score", fontsize=label_size)
    def rgb(r: int, g: int, b: int) -> str:
        return "#%02x%02x%02x" % (r, g, b)


    c_color = rgb(128, 0, 128)
    c_color_2 = rgb(252, 240, 0)
 
    alpha = 0.1

    styles = {
        "moo_points": dict(s=15, marker="o", color="red"),
        "moo_pareto": dict(s=4, marker="o", color="red", linestyle="-", linewidth=2),
        "so_points": dict(s=15, marker="o", color="black"),
        "so_pareto": dict(s=4, marker="o", color="black", linestyle="-", linewidth=2),
        "moo+cr_points": dict(s=15, marker="o", color="blue"),
        "moo+cr_pareto": dict(s=4, marker="o", color="blue", linestyle="-", linewidth=2),
        "moo_ps_points": dict(s=15, marker="o", color ="green"),
        "moo_ps_pareto": dict(s=4, marker="o", color="green", linestyle="-", linewidth=2),
        "moo+ps+cr_points": dict(s=15, marker="o", color =c_color),
        "moo+ps+cr_pareto": dict(s=4, marker="o", color=c_color, linestyle="-", linewidth=2),
        "moo+ps*cr_points": dict(s=15, marker="o", color=c_color_2),
        "moo+ps*cr_pareto": dict(s=4, marker="o", color=c_color_2, linestyle="-", linewidth=2),
    }
    for i,constrain in enumerate(data.keys()):
        global_max_y = 0
        global_min_y = 1
        for j,dataset in enumerate(data[constrain].keys()):
            print(dataset)
            global_min_x = 1
            global_max_x = 0
            #if len(data) ==1:
               # ax= axis 
            #else:
            if len(data.keys()) == 1:
                ax = axis[j]
            else:
                ax = axis[i,j]
            
            ax.set_title(dataset, fontsize=title_size)
            moo_pf = []
            so_pf = []
            moo_cr_pf = []
            moo_ps_pf = []
            moo_cr_ps_pf = []
            moo_cr_xor_ps_pf =[]
            max_len, max_len_ps, max_len_rl = 0,0,0
            for seed in data[constrain][dataset].keys():
                moo_pf.append(np.array(data[constrain][dataset][seed]['moo']['points']))
                #so_pf.append(np.array(data[constrain][dataset][seed]['so']['points']))
                moo_ps_pf.append(np.array(data[constrain][dataset][seed]['moo_ps_ranker']['points']))
                moo_cr_pf.append(np.array(data[constrain][dataset][seed]['moo+cr']['points']))
                moo_cr_ps_pf.append(np.array(data[constrain][dataset][seed]['ps_ranker']['points']))
                #moo_cr_xor_ps_pf.append(np.array(data[constrain][dataset][seed]['moo+ps*cr']['points']))

                #seed = "25" 
                #moo
                length = len(data[constrain][dataset][seed]['moo']['points'])
                max_len = length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['moo']['points'], ax=ax, **styles["moo_points"], alpha = alpha)

                #ps 
                #length = len(data[constrain][dataset][seed]['so']['points'])
                #max_len= length if max_len < length else max_len
                #plot(data[constrain][dataset][seed]['so']['points'], ax=ax, **styles["so_points"], alpha = alpha)

                #moo_ps
                length = len(data[constrain][dataset][seed]['moo_ps_ranker']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['moo_ps_ranker']['points'], ax=ax, **styles["moo_ps_points"], alpha = alpha)
               
                #moo_cr
                length = len(data[constrain][dataset][seed]['moo+cr']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['moo+cr']['points'], ax=ax, **styles["moo+cr_points"], alpha = alpha)

                #moo_cr_ps
                length = len(data[constrain][dataset][seed]['ps_ranker']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['ps_ranker']['points'], ax=ax, **styles["moo+ps+cr_points"], alpha = alpha)

                #moo_cr_xor_ps
                #length = len(data[constrain][dataset][seed]['moo+ps*cr']['points'])
                #max_len= length if max_len < length else max_len
                #plot(data[constrain][dataset][seed]['moo+ps*cr']['points'], ax=ax, **styles["moo+ps*cr_points"], alpha = alpha)


            for  i in range(len(moo_pf)):
                #moo
                diff = max_len-len(moo_pf[i]) 
                if diff:
                    moo_pf[i] = np.vstack((moo_pf[i], [moo_pf[i][-1]]*(diff)))
                #ps
                #diff = max_len-len(so_pf[i]) 
                #if diff:
                #    so_pf[i] = np.vstack((so_pf[i], [so_pf[i][-1]]*(diff)))

                #moo_ps
                diff = max_len-len(moo_ps_pf[i]) 
                if diff:
                    moo_ps_pf[i] = np.vstack((moo_ps_pf[i],[moo_ps_pf[i][-1]]*(diff)))
                #moo+cr
                diff = max_len-len(moo_ps_pf[i]) 
                if diff:
                    moo_ps_pf[i] = np.vstack((moo_ps_pf[i], [moo_ps_pf[i][-1]]*(diff)))
                #moo_ps
                diff = max_len-len(moo_cr_ps_pf[i]) 
                if diff:
                    moo_cr_ps_pf[i] = np.vstack((moo_cr_ps_pf[i],[moo_cr_ps_pf[i][-1]]*(diff)))
                #moo+cr
                #diff = max_len-len(moo_cr_xor_ps_pf[i]) 
                #if diff:
                #    moo_cr_xor_ps_pf[i] = np.vstack((moo_cr_xor_ps_pf[i], [moo_cr_xor_ps_pf[i][-1]]*(diff)))


            #moo_points = data[constrain][dataset][seed]['moo']['points']
            #ps_points = data[constrain][dataset][seed]['ps']['points']
            #redlineing_points = data[constrain][dataset][seed]['redlineing']['points']
            #moo_pf = data[constrain][dataset][seed]['moo']['points']
            #ps_pf = data[constrain][dataset][seed]['ps']['points']
            #redlineing_pf = data[constrain][dataset][seed]['redlineing']['points']
            #TODO: transform in shape(seed, point, metric)
            #TODO: almoo_ps need to thing about that not every seed has the same amount of points
            #moo_pf = np.vstack(moo_pf)
            #t = copy.deepcopy(ax)
            #_, ax = plt.subplots(ax)
            # ,np.stack(redlineing_pf,axis=0)
            # np.stack(moo_pf, axis=0),np.stack(moo_cr_pf, axis=0),np.stack(moo_ps_pf, axis=0),np.stack(ps_pf, axis=0)
            pf = [np.stack(moo_pf, axis=0),
                #np.stack(so_pf, axis=0),
                np.stack(moo_cr_pf, axis=0),
                np.stack(moo_ps_pf, axis=0),
                np.stack(moo_cr_ps_pf, axis=0),
                #np.stack(moo_cr_xor_ps_pf, axis=0)
                ]
            # , styles["redlineing_points"]['color'] , styles["moo_ps_pareto"]['color'],styles["ps_pareto"]['color']
            plots_we(pf, ax, [styles["moo_pareto"]['color'],
                              #styles["so_pareto"]['color'],
                              styles["moo+cr_pareto"]['color'],
                              styles["moo_ps_pareto"]['color'], 
                              styles["moo+ps+cr_pareto"]['color'],
                              #styles["moo+ps*cr_pareto"]['color']
                              ])
            ax.tick_params(axis="both", which="major", labelsize=tick_size)
    legend_elements = [
        Line2D([0], [0], color=styles["moo_pareto"]['color'], lw=4, label='moo'),
        #Line2D([0], [0], color=styles["so_pareto"]['color'], lw=4, label='single objective optimization, but with accuracy'),
        Line2D([0], [0], color=styles["moo+cr_pareto"]['color'],lw=4, label='moo with optinionall cr'),
        Line2D([0], [0], color=styles["moo_ps_pareto"]['color'], lw=4, label='moo and optinionall sampling'),
        Line2D([0], [0], color=styles["moo+ps+cr_pareto"]['color'], lw=4, label='moo with only sampling'),
        #Line2D([0], [0], color=styles["moo+ps*cr_pareto"]['color'], lw=4, label='moo with optinionall sampling xor cr')
                    ]
    fig.tight_layout(rect=[0.03, 0.05, 1, 1], pad = 5)
    fig.legend(handles=legend_elements, loc=3,  prop={'size': 16})
    #save_folder = "/home/till/Desktop/all_prepreprocessor_plots/sampling+cr/{}".format("error_rate_difference_all")
    save_folder = "/home/till/Desktop/{}".format("consistency_score_all")
    plt.savefig(save_folder)
def plot_arrows(
        to,
        frm,
        ax,
        **kwargs,
    ):
       

        if ax is None:
            ax = plt.gca()

        default = {
            "color": "grey",
            "width": 0.01,
        }
        kwargs = {**default, **kwargs}

       

        # Define a list of segments going from one point to the other
        frm = pd.DataFrame(frm)
        dif =  frm - to
        
        ax.quiver(to[0], to[1], dif[0], dif[1], angles="xy", scale_units="xy", scale=1, **kwargs)

        return 

def right_alpha(data, alpha_ps):
    points = []
    configs = []
    h_points = []
    
    for idx, conf in enumerate(data["configs"]):
        if idx == 0:
            continue
        if conf['feature_preprocessor:CorrelationRemover:alpha'] == 0.0:
            continue
        if alpha_ps == "all":
            
            return data['points'][1:], pd.DataFrame(data["configs"][1:])
        elif alpha_ps == "best":
            h_points.append(np.array(data['points'][idx:(idx+1)]))
            if len(h_points)==10:
                idx_best = np.argmin(np.squeeze(np.stack(h_points, axis=0)),axis=0)[1]#
                points.append(h_points[idx_best]) #min of fairness
                h_points = []
        else:
            # because  
            if conf['feature_preprocessor:CorrelationRemover:alpha'] == int(alpha_ps/0.1)*0.1:        
                points.append(np.array(data['points'][idx:(idx+1)]))
                configs.append(data['configs'][idx:(idx+1)][0])
    
    try:
        points = np.stack(points, axis=0)
    except:
        print()
    points = np.squeeze(points)
    return points, pd.DataFrame(configs)
import itertools
def calc_index(conf):
    # in: conf, num of points on pareto
    # do: but the right points on the pareto
    # out: give the indexies back.
    similar_rows = []
    groups = {}
    for idx, row in conf.iterrows():
        hyperparams = tuple(row.drop(["classifier:random_forest:random_state_forest", "feature_preprocessosr:CorrelationRemover:alpha"]))
        if hyperparams not in groups:
            groups[hyperparams] = [idx]
        else:
            groups[hyperparams].append(idx)
    similar_rows = [value for value in groups.values()]
    max_len = max(len(lst) for lst in similar_rows)
    for lst in similar_rows:
        if len(lst) < max_len:
            lst.extend([lst[-1]] * (max_len - len(lst)))

    result = []
    for i, hyperparams in enumerate(groups.keys()):
        row_indices = similar_rows[i]
        row_indices = [row_indices[0]] + [idx for idx in row_indices if idx != row_indices[0]]
        row_indices.extend([row_indices[-1]] * (max_len - len(row_indices)))
        result.append(row_indices)
    result = list(map(list, zip(*result)))

    return result






def make_difference_plot(data, alpha_ps):
    # in: data should have the pareto_front and the results of alpha  
    # do: check of the different alpha the best one, if their are multiple best the one with the highest fairness
    # out: image with the difference between pareto front and the ps output. 
        
        sns.set_context("paper", font_scale=0.6)

        #TODO set on big monitor
        figsize = (27,10)
        dpi = 300
        main_size = 20
        plot_offset = 0.1
        title_size = 18
        label_size = 16
        tick_size = 12
    
    
        
        fig, axis = plt.subplots(
            nrows=len(data),
            ncols=len(data[list(data.keys())[0]]),  #needs to be more flexible for nowe is ok
            #sharey=True,
            figsize=figsize,
        )
        
        fig.supxlabel("error",fontsize=label_size)
        fig.supylabel("consistency_score", fontsize=label_size)
        def rgb(r: int, g: int, b: int) -> str:
            return "#%02x%02x%02x" % (r, g, b)


        c_color = rgb(128, 0, 128)
    
        alpha = 1

        styles = {
            "moo_points": dict(s=15, marker="o", color="red"),
            "moo_pareto": dict(s=4, marker="o", color="red", linestyle="-", linewidth=2),
            "ps_points": dict(s=15, marker="o", color="blue"),
            "ps_pareto": dict(s=4, marker="o", color="blue", linestyle="-", linewidth=2),
        }
        for i,constrain in enumerate(data.keys()):
            global_max_y = 0
            global_min_y = 1
            for j,dataset in enumerate(data[constrain].keys()):
                #dataset = "german"
                print(dataset)
                global_min_x = 1
                global_max_x = 0
                global_max_y = 0
                global_min_y = 1
                if len(data[constrain].keys()) ==1:
                    ax= axis 
                else:
                    if len(data.keys()) == 1:
                        ax = axis[j]
                    else:
                        ax = axis[i,j]
                
                ax.set_title(dataset, fontsize=title_size)
                moo_pf = []
                ps_pf = []
                redlineing_pf = []
                #lfr_pf = []
                max_len, max_len_ps, max_len_rl = 0,0,0
                for seed in data[constrain][dataset].keys():
                    #seed = "42" 
                    moo_pf.append(np.array(data[constrain][dataset][seed]['moo']['pareto_set']))
                    ps_front, conf = right_alpha(data[constrain][dataset][seed]['ps'], alpha_ps)
                    ps_pf.append(np.array(ps_front))
                    #lfr_pf.append(np.array(data[constrain][dataset][seed]['lfr']['points']))
                    
                    #moo
                    length = len(data[constrain][dataset][seed]['moo']['pareto_set'])
                    max_len = length if max_len < length else max_len
                    #plot(data[constrain][dataset][seed]['moo']['points'], ax=ax, **styles["moo_points"], alpha = alpha)

                    #ps 
                    length = len(data[constrain][dataset][seed]['ps']['points'])
                    max_len= length if max_len < length else max_len
                    #plot(data[constrain][dataset][seed]['ps']['points'], ax=ax, **styles["ps_points"], alpha = alpha)
                    #pareto_plot(data[constrain][dataset][seed]['moo']['pareto_set'], ax=ax, **styles["moo_pareto"])
                    #pareto_plot(pd.DataFrame(ps_pf[-1]), ax=ax, **styles["ps_pareto"])
                    #variants = max(ps_front
                    
                    #indecies = calc_index(conf)
                    #for index in indecies:
                        #indexes = [(j*(seeds+diff_alphas)) + i  for j in range(0,len(data[constrain][dataset][seed]['moo']['pareto_set']))]
                    
                    if len(ps_pf[-1].shape)==1:
                        ps_pf[-1] = [ps_pf[-1]]
                    #plot_arrows(data[constrain][dataset][seed]['moo']['pareto_set'],ps_pf[-1],ax)
                    ps = pd.DataFrame(ps_pf[-1])
                    local_min_x = min(min(data[constrain][dataset][seed]['moo']['pareto_set'][0]), min(ps[0]))
                    local_min_y = min(min(data[constrain][dataset][seed]['moo']['pareto_set'][1]), min(ps[1]))
                    local_max_x = max(max(data[constrain][dataset][seed]['moo']['pareto_set'][0]), max(ps[0]))
                    local_max_y = max(max(data[constrain][dataset][seed]['moo']['pareto_set'][1]), max(ps[1]))
                    #local_min_x = min(min(moo_points[0]), min(ps_points[0]), min(redlineing_points[0]))
                    #local_min_y = min(min(moo_points[1]), min(ps_points[1]), min(redlineing_points[1]))
                    #local_max_x = max(max(moo_points[0]), max(ps_points[0]), max(redlineing_points[0]))
                    #local_max_y = max(max(moo_points[1]), max(ps_points[1]), max(redlineing_points[1]))
                    global_min_y = local_min_y if local_min_y < global_min_y  else global_min_y
                    global_min_x = local_min_x if local_min_x < global_min_x  else global_min_x
                    global_max_y = local_max_y if local_max_y > global_max_y  else global_max_y
                    global_max_x = local_max_x if local_max_x > global_max_x  else global_max_x
                dx = abs(global_max_x - global_min_x) if abs(global_max_x - global_min_x) > 0 else 0.01
                ax.set_xlim(max(global_min_x - dx * plot_offset,0), global_max_x + dx * plot_offset)
                dy = abs(global_max_y - global_min_y)
                ax.set_ylim(max(global_min_y - dy*plot_offset,0), global_max_y +  dy * plot_offset)
                for  i in range(len(moo_pf)):
                    #moo
                    diff = max_len-len(moo_pf[i]) 
                    if diff:
                        try:
                            moo_pf[i] = np.vstack((moo_pf[i], [moo_pf[i][-1]]*(diff)))
                        except:
                            moo_pf[i] = np.vstack(([moo_pf[i]]*(max_len)))   
                    #ps
                    diff = max_len-len(ps_pf[i]) 
                    if diff:
                        try:
                            ps_pf[i] = np.vstack((ps_pf[i], [ps_pf[i][-1]]*(diff)))
                        except:
                            ps_pf[i] = np.vstack(([ps_pf[i]]*(max_len)))


                
                pf = [np.stack(moo_pf, axis=0), np.stack(ps_pf, axis=0)]
                #pareto_plot(moo_pf, ax=ax, **styles["moo_pareto"])
                #pareto_plot(ps_pf, ax=ax, **styles["ps_pareto"])
                plots_we(pf, ax, [styles["moo_points"]['color'],styles["ps_points"]['color']])

                ax.tick_params(axis="both", which="major", labelsize=tick_size)
        legend_elements = [
                    Line2D([0], [0], color="red", lw=4, label='moo without preprocessing'),
                    Line2D([0], [0], color="blue", lw=4, label='moo with correlation remover'),
                    #Line2D([0], [0], color="green", lw=4, label='moo without SA and corrleation remover'),
                        #Line2D([0], [0], color=c_color, lw=4, label='moo with learned fair represenation')
                        ]
        fig.tight_layout(rect=[0.03, 0.05, 1, 1], pad = 5)
        fig.legend(handles=legend_elements, loc=3,  prop={'size': 16})
        save_folder = "/home/till/Desktop/arrows/{}/seed{}".format(constrain," all")
        plt.savefig(save_folder)
def complex_xor_preproecessing(point):
    both = 0
    preprocessor = "no"
    try:
        if point["fair_preprocessor:__choice__"] != "NoFairPreprocessor":
            both += 1
            preprocessor = point["fair_preprocessor:__choice__"]
    except KeyError:
        pass
    if point["feature_preprocessor:__choice__"] != "no_preprocessing":
        both += 1
        preprocessor = point["feature_preprocessor:__choice__"]
    return preprocessor if both < 2 else "both"
    




def make_choice_file(data, file, methods):
    prepreossor_dict = defaultdict()
    for i,constrain in enumerate(data.keys()):
        prepreossor_dict[constrain] = defaultdict()
        for j,dataset in enumerate(data[constrain].keys()):
            prepreossor_dict[constrain][dataset] = defaultdict()
            for seed in data[constrain][dataset].keys():
                prepreossor_dict[constrain][dataset][seed] = defaultdict()
                prepreossor_dict[constrain][dataset][seed]["preprocessor"] = []
                for method in methods:
                    prepreossor_dict[constrain][dataset][seed][method] = defaultdict()
                    prepreossor_dict[constrain][dataset][seed][method]["preprocessor"] = []
                    prepreossor_dict[constrain][dataset][seed][method]["preprocessorp"] = []
                    prepreossor_dict[constrain][dataset][seed][method]["preprocessor_hyper"] = []
                    points = data[constrain][dataset][seed][method]['pareto_config']
                    for point in points:
                        prepreossor_dict[constrain][dataset][seed][method]["preprocessor"].append(complex_xor_preproecessing(point))
                        #prepreossor_dict[constrain][dataset][seed][method]["preprocessor"].append(point["fair_preprocessor:__choice__"])
                    h_list =  prepreossor_dict[constrain][dataset][seed][method]["preprocessor"]
                    prepreossor_dict[constrain][dataset][seed][method]["percentage_use"] = {k: (v / len(h_list)) * 100 for k, v in dict(zip(set(h_list), map(h_list.count, set(h_list)))).items()}            
    with open(file, 'w') as f:
        json.dump(prepreossor_dict, f, indent=4)
    # iterate over each combination of dataset and constraint
    
    for constrain in prepreossor_dict.keys():
        for dataset in prepreossor_dict[constrain].keys():
            total_percentage_cr = 0
            total_percentage_ps = 0
            total_percentage_b = 0
            count = 0
            
            # iterate over each entry in the prepreossor_dict dictionary
            for seed in prepreossor_dict[constrain][dataset].keys():
                count += 1
                if  "CorrelationRemover" in  prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use'].keys():
                    total_percentage_cr += prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use']["CorrelationRemover"]
                if "PreferentialSampling" in  prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use'].keys():
                    total_percentage_ps += prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use']['PreferentialSampling']
                if "both" in  prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use'].keys():
                    total_percentage_b += prepreossor_dict[constrain][dataset][seed][methods[0]]['percentage_use']['both']       
            
            # calculate the average percentage of correlation remover use
          
            average_percentage_cr = total_percentage_cr / count
            average_percentage_ps = total_percentage_ps / count
            average_percentage_b = total_percentage_b / count
           
            
            # print the result
            print(f"{dataset} + {constrain}: {average_percentage_ps} sampling %")
            print(f"{dataset} + {constrain}: {average_percentage_cr} cr %")
            print(f"{dataset} + {constrain}: {average_percentage_b} % b")
from pymoo.indicators.hv import HV
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
def calc_hypervolume(data,file):
    scaler = MinMaxScaler()
    hypervolume_obj =  HV(ref_point=np.array([1,1]))
    hypervolume_dict = defaultdict()
    for constrain in data.keys():
        hypervolume_dict[constrain] = defaultdict()
        for dataset in data[constrain].keys():
            hypervolume_dict[constrain][dataset] = defaultdict()
            hypervolume_dict[constrain][dataset]["hypervolume"] = []
            hypervolume_dict[constrain][dataset]["fairness"] = []
            hypervolume_dict[constrain][dataset]["acc"] = []
            
            hypervolume_dict[constrain][dataset]["methods"] = [method for method in data[constrain][dataset]["97"].keys()]
            help_dict_hv = defaultdict()
            help_dict_f = defaultdict()
            help_dict_a = defaultdict()
            for seed in data[constrain][dataset].keys():
                for method in data[constrain][dataset][seed].keys():
                    if not(method in help_dict_hv.keys()):
                            help_dict_hv[method] = []
                            help_dict_f[method] = []
                            help_dict_a[method] = []
                    pareto_front = data[constrain][dataset][seed][method]["pareto_set"]
                    
                    help_dict_hv[method].append(hypervolume_obj(np.array(pareto_front)))
                    help_dict_f[method].append(mean(pareto_front[1]))
                    help_dict_a[method].append(mean(pareto_front[0]))
                    if len( help_dict_hv[method]) == len(data[constrain][dataset].keys()):
                        hypervolume_dict[constrain][dataset]["hypervolume"].append(mean(help_dict_hv[method]))
                        hypervolume_dict[constrain][dataset]["fairness"].append(mean(help_dict_f[method]))
                        hypervolume_dict[constrain][dataset]["acc"].append(mean(help_dict_a[method]))

            hv = np.array(hypervolume_dict[constrain][dataset]["hypervolume"]).reshape(-1,1)
            fairness = np.array(hypervolume_dict[constrain][dataset]["fairness"]).reshape(-1,1)
            accurancy = np.array(hypervolume_dict[constrain][dataset]["acc"]).reshape(-1,1)
            scaled_hv = scaler.fit_transform(hv).tolist()
            scaled_fairness = scaler.fit_transform(fairness).tolist()
            scaled_acc = scaler.fit_transform(accurancy).tolist()
            hypervolume_dict[constrain][dataset]["hypervolume_scaled_max"] = [value[0] for value in scaled_hv]
            hypervolume_dict[constrain][dataset]["fairness_scaled_max"] = [1-value[0] for value in scaled_fairness]
            hypervolume_dict[constrain][dataset]["acc_scaled_max"] = [1-value[0] for value in scaled_acc]
    print(hypervolume_dict)
    with open(file, 'w') as f:
        json.dump(hypervolume_dict, f, indent=4)
def plot_scaled_values(results,result_folder,plot_feature, offset=0.1, scale_type='0-1', one_line=False, label=False):
    def rgb(r: int, g: int, b: int) -> str:
        return "#%02x%02x%02x" % (r, g, b)
    c_color = rgb(128, 0, 128)
    c_color_2 = rgb(252, 240, 0)
    c_color_3 = rgb(0,128, 128)
    c_color_4 = rgb(128,128,0)
    styles = {
        "moo": dict(s=15, marker="o", color="red"),
        "so": dict(s=15, marker="o", color="black"),
        "moo+cr": dict(s=15, marker="o", color="blue"),
        "moo_ps_ranker": dict(s=15, marker="o", color ="green"),
        "moo+ps+cr": dict(s=15, marker="o", color =c_color),
        "moo+ps*cr": dict(s=15, marker="o", color=c_color_2),
        "cr":  dict(s=15, marker="o", color=c_color_3),
        "ps": dict(s=15, marker="o", color=c_color_4),
    }
    for constrain in results.keys():
    
        if not one_line:
            plot_width = .7 * len(results)
        else:
            plot_width = .42 * len(results)        
        fig, ax = plt.subplots(figsize=(plot_width,2))
        width = 0
        ss_order = results[constrain].keys()
        for dataset in results[constrain].keys():
            width += 1
            for idx, method in enumerate(results[constrain][dataset]["methods"]):
                color = styles[method]['color']
                data = results[constrain][dataset][plot_feature][idx]
                if width == 1:
                    m = method if method != "moo_ps_ranker" else "moo_ps"
                    ax.scatter(width, data, label=m, color=color, marker='o')
                else:
                    ax.scatter(width, data, color=color, marker='o')

            
        

        ax.set_xticks(range(1, len(ss_order)+1))
        
        ax.set_xticklabels(results[constrain].keys(), fontsize=12, rotation=90)
       

        if one_line:
            ax.legend(loc=(1.5,-.6))
        else:
            ax.legend(loc=(1.01,0))
        #ax.set_xlabel('NAS Benchmark Task', fontsize=14)    
        ax.set_ylabel(plot_feature, fontsize=12)   
        #ax.set_title('Scaled Accuracy of NAS Algorithms', fontsize=14)
        c = constrain if "hypervolume" not in plot_feature else "Accuracy x \n"+constrain
        ax.set_title(c, fontsize=14)
        
        # ax.set_ylim([0.87, 1.01])

        
        plt.savefig(result_folder + plot_feature +  "_" + constrain, bbox_inches = 'tight', pad_inches = 0.1)
        #print(results)    
def get_possible_pre(method):
    if method == "moo+cr":
        return ["no", "CorrelationRemover"]
    if method == "moo_ps_ranker":
         return ["no", "PreferentialSampling"]
    if method == "moo+ps+cr":
        return ["no", "PreferentialSampling","CorrelationRemover"]
    if method == "moo+ps*cr":
        return ["no", "PreferentialSampling","CorrelationRemover", "both"]

def calc_pareto_contribution(data,file, methods):
    scaler = MinMaxScaler()
    contribution = defaultdict()
    for i,constrain in enumerate(data.keys()):
        contribution[constrain] = defaultdict()
        for j,dataset in enumerate(data[constrain].keys()): 
            contribution[constrain][dataset] = defaultdict()
            for seed in data[constrain][dataset].keys():
                for method in methods:
                    if method not in  contribution[constrain][dataset].keys():
                        contribution[constrain][dataset][method] = defaultdict()
                        contribution[constrain][dataset][method]["run"] = 0
                    contribution[constrain][dataset][method]["run"] += 1
                    preprocessor_dict = defaultdict()
                    
                    pareto_set = data[constrain][dataset][seed][method]["pareto_set"]
                    acc = np.array(pareto_set[0]).reshape(-1,1)
                    fairness =  np.array(pareto_set[1]).reshape(-1,1)
                    acc = scaler.fit_transform(acc) if len(pareto_set)>1 else np.array([1])
                    fairness = scaler.fit_transform(fairness) if len(pareto_set)>1 else np.array([1])
                    if len(pareto_set)<=1:
                        print("constrain:{}, dataset:{}, method:{}".format(constrain, dataset, method))
                    A = sum(acc)
                    F = sum(fairness)
                    points = data[constrain][dataset][seed][method]['pareto_config']
                    for idx,point in enumerate(points):
                        pre = complex_xor_preproecessing(point)
                        if pre not in preprocessor_dict.keys():
                            preprocessor_dict[pre] = [0,0]
                        preprocessor_dict[pre][0] += float(acc[idx])
                        preprocessor_dict[pre][1] += float(fairness[idx])
                    #scale all weights
                    possible_pre = get_possible_pre(method)
                    for pre in possible_pre:
                       
                        if pre not in preprocessor_dict.keys():
                            preprocessor_dict[pre] = [0,0]
                        try:
                            preprocessor_dict[pre][0] /=  float(A)
                        except ZeroDivisionError:
                            preprocessor_dict[pre][0] = 0
                        try:
                            preprocessor_dict[pre][1] /= float(F)
                        except ZeroDivisionError:
                            preprocessor_dict[pre][1] = 0
                        if pre not in contribution[constrain][dataset][method].keys():
                            contribution[constrain][dataset][method][pre] = defaultdict()
                            contribution[constrain][dataset][method][pre]["acc"] = []
                            contribution[constrain][dataset][method][pre]["fairness"] = []
                            contribution[constrain][dataset][method][pre]["both"] = []
                        #print("constrain:{}, dataset:{}, method:{}, run:{}".format(constrain,dataset, method,contribution[constrain][dataset][method]["run"] ))
                        contribution[constrain][dataset][method][pre]["acc"].append(preprocessor_dict[pre][0])
                        contribution[constrain][dataset][method][pre]["fairness"].append(preprocessor_dict[pre][1])
                        contribution[constrain][dataset][method][pre]["both"].append((preprocessor_dict[pre][0]+preprocessor_dict[pre][1])/2)
            no_seeds = len(data[constrain][dataset].keys())       
            for method in methods:
                possible_pre = get_possible_pre(method)
                for pre in possible_pre:
                    contribution[constrain][dataset][method][pre]["acc"] = sum(contribution[constrain][dataset][method][pre]["acc"])/no_seeds
                    contribution[constrain][dataset][method][pre]["fairness"] = sum(contribution[constrain][dataset][method][pre]["fairness"])/no_seeds
                    contribution[constrain][dataset][method][pre]["both"] = sum(contribution[constrain][dataset][method][pre]["both"])/no_seeds
                    
    with open(file, 'w') as f:
        json.dump(contribution, f, indent=4)
    return  
                    

                    #print(preprocessor_dict)

                            



if __name__ == "__main__":
    #methods = ["moo_ps_ranker","moo","moo+cr", "ps_ranker"]
    
    #data = load_data("/home/till/Documents/auto-sklearn/tmp/cross_val/", "200timesstrat", methods)
    #make_plot_3(data)
    #data = load_data_particully("/home/till/Desktop/psoss_val/", "200timesstrat",
    #datasets = ["german","adult", "compass","lawschool"],
    #constrains = ["consistency_score"],
    #folders=["one_rf_seed_1", "one_rf_seed"],
    #"12345","25","42","45451", "97","13","27","39","41","53"
    seeds= ["12345","25","42","45451", "97","13","27","39","41","53"]
    #make_difference_plot(data,"best")
    #methods = ["moo","so","cr","ps","moo_ps_ranker","moo+cr","moo+ps+cr","moo+ps*cr"]
    methods = ["moo+cr","moo+ps+cr","moo+ps*cr"]
    data = load_data("/home/till/Documents/auto-sklearn/tmp/cross_val/", "200timesstrat", methods)
    deep_dive = defaultdict()
    for method in methods:
        for seed in seeds:
            deep_dive[seed] = data["error_rate_difference"]["adult"][seed][method]["pareto_set"].to_json()
        file = "/home/till/Documents/auto-sklearn/tmp/deep_dive_error_rate_difference_values_adult_{}.json".format(method)
        with open(file, 'w') as f:
            json.dump(deep_dive, f, indent=4)
    #with open(file) as f:
    #    results = json.load(f)
    #calc_hypervolume(data, file)
    #plot_scaled_values(results,"/home/till/Documents/auto-sklearn/tmp/",'hypervolume_scaled_max')
    #make_choice_file(data, file, methods)