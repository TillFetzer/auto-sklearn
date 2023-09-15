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
                        file  = "{}/{}/{}/runhistory.json".format(seed_path,method,"200timestrat")
                    if not(os.path.exists(file)):
                        file  = "{}/{}/{}/del/smac3-output/run_{}/runhistory.json".format(seed_path,method,runetime_folder,seed) 
                    if not(os.path.exists(file)):   
                            file  = "{}/{}/{}/runhistory.json".format(seed_path,method,"150timestrat")
                    if not(os.path.exists(file)):
                         if not(os.path.exists(file)):   
                            file  = "{}/{}/{}/runhistory.json".format(seed_path,method,"150timesstrat") 
                    if not(os.path.exists(file)):
                        print(file)    
                        #else:
                        #    print(file)   
                    #else:
                    #    print(file)             
            
                                    
                    with open(file) as f:
                        ds = json.load(f)
                    for d in ds["data"]:
                        try:
                            if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                                continue
                           
                            point = d[1][0]
                            config = ds["configs"][str(d[0][0])]
                            if method == "so" or method == "so_lfr":
                                p = []
                                p.append(point)
                                if method == "so":
                                    ff  = "{}/{}/{}/fairness.json".format(seed_path,method,runetime_folder)
                                    with open(ff) as f:
                                        fairness = json.load(f)
                                    if constrain == "consistency_score":
                                        p.append(1-fairness[d[0][0] -1])
                                    else:
                                        p.append(fairness[d[0][0] -1])
                                elif method == "so_lfr":
                                    p.append(list(d[1][5].values())[0])

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
def pareto_set(all_costs):

    confs = all_costs["configs"]
    all_costs = np.array(all_costs["points"])
    assert len(all_costs) == len(confs)
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

def pareto_points(all_costs):
    all_costs = np.array(all_costs)
    moo_psrt_by_first_metric = np.argsort(all_costs[:, 0])
    efficient_points = pareto_front(all_costs, is_loss=True)
    pareto_set = []
    for argmoo_psrt_idx in moo_psrt_by_first_metric:
            if not efficient_points[argmoo_psrt_idx]:
                continue
            pareto_set.append(all_costs[argmoo_psrt_idx,:])            
            
    
    return pareto_set


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
    fig.supylabel("1-equalized_odds", fontsize=label_size)
    def rgb(r: int, g: int, b: int) -> str:
        return "#%02x%02x%02x" % (r, g, b)


    c_color = rgb(128, 0, 128)
    c_color_2 = rgb(252, 240, 0)
 
    alpha = 0.1

    styles = {
        "moo": dict(s=15, marker="o", color="red"),
        #"so": dict(s=15, marker="o", color="green"),
        "ps_ranker": dict(s=15, marker="o", color="blue"),
        "cr": dict(s=15, marker="o", color ="green"),
        "lfr": dict(s=15, marker="o", color=c_color_2),
        "so":  dict(s=15, marker="o", color="black"),
        "redlineing": dict(s=15, marker="o", color=c_color),
    }
    for i,constrain in enumerate(data.keys()):
      
        for j,dataset in enumerate(data[constrain].keys()):
            print(dataset)
            
            #else:
            if len(data.keys()) == 1:
                ax = axis[j]
            else:
                ax = axis[i,j]
            
            ax.set_title(dataset, fontsize=title_size)
            moo_pf = []
            moo_ps_pf = []
            moo_ps_xor_cr_pf = []
            op_pf = []
            moo_ps_lfr_pf = []
            all_pf = []
            moo_ps_cr_pf = []
            moo_ps_cr_pf = []
            max_len  = 0
            
            for seed in data[constrain][dataset].keys():
                moo_pf.append(np.array(data[constrain][dataset][seed]['moo']['points']))
                moo_ps_xor_cr_pf.append(np.array(data[constrain][dataset][seed]['cr']['points']))
                #op_pf.append(np.array(data[constrain][dataset][seed]['ps+cr+lfr']['points']))
                moo_ps_cr_pf.append(np.array(data[constrain][dataset][seed]['ps_ranker']['points']))
                moo_ps_lfr_pf.append(np.array(data[constrain][dataset][seed]['lfr']['points']))
                all_pf.append(np.array(data[constrain][dataset][seed]['redlineing']['points']))
              

                #moo
                length = len(data[constrain][dataset][seed]['moo']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['moo']['points'], ax=ax, **styles["moo"], alpha = alpha)

                #moo+ps+cr
                length = len(data[constrain][dataset][seed]['cr']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['cr']['points'], ax=ax, **styles["cr"], alpha = alpha)

                #ps+cr+lfr
                length = len(data[constrain][dataset][seed]['ps_ranker']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['ps_ranker']['points'], ax=ax, **styles["ps_ranker"], alpha = alpha)

                #moo+ps*cr
                #length = len(data[constrain][dataset][seed]['lfr']['points'])
                #max_len= length if max_len < length else max_len
                #plot(data[constrain][dataset][seed]['lfr']['points'], ax=ax, **styles["lfr"], alpha = alpha)

                #moo+ps+lfr
                length = len(data[constrain][dataset][seed]['redlineing']['points'])
                max_len= length if max_len < length else max_len
                plot(data[constrain][dataset][seed]['redlineing']['points'], ax=ax, **styles["redlineing"], alpha = alpha)

                #moo+ps+cr+lfr
                #length = len(data[constrain][dataset][seed]['moo+ps+cr+lfr']['points'])
                #max_len= length if max_len < length else max_len
                #plot(data[constrain][dataset][seed]['moo+ps+cr+lfr']['points'], ax=ax, **styles["lfr"], alpha = alpha)



            for  i in range(len(moo_pf)):
                #moo
                diff = max_len-len(moo_pf[i]) 
                if diff:
                    moo_pf[i] = np.vstack((moo_pf[i], [moo_pf[i][-1]]*(diff)))

                #moo+ps+cr  
                diff = max_len-len(moo_ps_xor_cr_pf[i])
                if diff:
                    moo_ps_xor_cr_pf[i] = np.vstack((moo_ps_xor_cr_pf[i], [moo_ps_xor_cr_pf[i][-1]]*(diff)))
                
                #ps+cr+lfr
                #diff = max_len-len(op_pf[i])
                #if diff:
                #    op_pf[i] = np.vstack((op_pf[i], [op_pf[i][-1]]*(diff)))
                
                #moo+ps*cr
                diff = max_len-len(moo_ps_cr_pf[i])
                if diff:
                    moo_ps_cr_pf[i] = np.vstack((moo_ps_cr_pf[i], [moo_ps_cr_pf[i][-1]]*(diff)))


                #moo+ps+lfr
                diff = max_len-len(moo_ps_lfr_pf[i])
                if diff:
                    moo_ps_lfr_pf[i] = np.vstack((moo_ps_lfr_pf[i], [moo_ps_lfr_pf[i][-1]]*(diff)))
                

                #moo+ps+cr+lfr
                diff = max_len-len(all_pf[i])
                if diff:
                    all_pf[i] = np.vstack((all_pf[i], [all_pf[i][-1]]*(diff)))
                



            pf = [
                np.stack(moo_pf, axis=0),
                np.stack(moo_ps_xor_cr_pf, axis=0),
                #np.stack(op_pf, axis=0),
                np.stack(moo_ps_cr_pf, axis=0),
                #np.stack(moo_ps_lfr_pf, axis=0),
                np.stack(all_pf, axis=0)
                ]
            # , styles["redlineing_points"]['color'] , styles["moo_ps_pareto"]['color'],styles["ps_pareto"]['color']
            plots_we(pf, ax, [
                              styles["moo"]['color'],
                              styles["cr"]['color'],
                              styles["ps_ranker"]['color'],
                              styles["redlineing"]['color'], 
                              #styles["lfr"]['color'],
                              ])
            ax.tick_params(axis="both", which="major", labelsize=tick_size)
    legend_elements = [
        Line2D([0], [0], color=styles["moo"]['color'], lw=4, label="Moo"),
        Line2D([0], [0], color=styles["cr"]['color'], lw=4, label='cr'),
        Line2D([0], [0], color=styles["ps_ranker"]['color'],lw=4, label='ps'),
        
        Line2D([0], [0], color=styles["redlineing"]['color'], lw=4, label='redlineing')
        
      
                   ]
    fig.tight_layout(rect=[0.03, 0.09, 1, 1], pad = 5)
    fig.legend(handles=legend_elements, loc=3,  prop={'size': 16})
    #save_folder = "/home/till/Desktop/all_prepreprocessor_plots/sampling+cr/{}".format("error_rate_difference_all")
    save_folder = "/home/till/Desktop/redlineing/{}".format("equalized_odds.png")
    #plt.show()
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
                moo_ps_pf = []
                redlineing_pf = []
                #lfr_pf = []
                max_len, max_len_ps, max_len_rl = 0,0,0
                for seed in data[constrain][dataset].keys():
                    #seed = "42" 
                    moo_pf.append(np.array(data[constrain][dataset][seed]['moo']['pareto_set']))
                    ps_front, conf = right_alpha(data[constrain][dataset][seed]['ps'], alpha_ps)
                    moo_ps_pf.append(np.array(ps_front))
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
                    #pareto_plot(pd.DataFrame(moo_ps_pf[-1]), ax=ax, **styles["ps_pareto"])
                    #variants = max(ps_front
                    
                    #indecies = calc_index(conf)
                    #for index in indecies:
                        #indexes = [(j*(seeds+diff_alphas)) + i  for j in range(0,len(data[constrain][dataset][seed]['moo']['pareto_set']))]
                    
                    if len(moo_ps_pf[-1].shape)==1:
                        moo_ps_pf[-1] = [moo_ps_pf[-1]]
                    #plot_arrows(data[constrain][dataset][seed]['moo']['pareto_set'],moo_ps_pf[-1],ax)
                    ps = pd.DataFrame(moo_ps_pf[-1])
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
                    diff = max_len-len(moo_ps_pf[i]) 
                    if diff:
                        try:
                            moo_ps_pf[i] = np.vstack((moo_ps_pf[i], [moo_ps_pf[i][-1]]*(diff)))
                        except:
                            moo_ps_pf[i] = np.vstack(([moo_ps_pf[i]]*(max_len)))


                
                pf = [np.stack(moo_pf, axis=0), np.stack(moo_ps_pf, axis=0)]
                #pareto_plot(moo_pf, ax=ax, **styles["moo_pareto"])
                #pareto_plot(moo_ps_pf, ax=ax, **styles["ps_pareto"])
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

    #for constrain in data.keys():
    for constrain in data.keys():
        hypervolume_dict[constrain] = defaultdict()
        for dataset in data[constrain].keys():
            hypervolume_dict[constrain][dataset] = defaultdict()
            hypervolume_dict[constrain][dataset]["hypervolume"] = []
            hypervolume_dict[constrain][dataset]["fairness"] = []
            hypervolume_dict[constrain][dataset]["acc"] = []
            hypervolume_dict[constrain][dataset]["fairness_best"] = []
            hypervolume_dict[constrain][dataset]["acc_best"] = []
            hypervolume_dict[constrain][dataset]["hypervolume_seed_range"] = []
            hypervolume_dict[constrain][dataset]["fairness_seed_range"] = []
            hypervolume_dict[constrain][dataset]["acc_seed_range"] = []
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
                        hypervolume_dict[constrain][dataset]["fairness"].append(1-mean(help_dict_f[method]))
                        hypervolume_dict[constrain][dataset]["acc"].append(1-mean(help_dict_a[method]))
                        hypervolume_dict[constrain][dataset]["acc_best"].append(1-min(help_dict_a[method]))
                        hypervolume_dict[constrain][dataset]["fairness_best"].append(1-min(help_dict_f[method]))
                        hypervolume_dict[constrain][dataset]["hypervolume_seed_range"].append(max(help_dict_hv[method])-min(help_dict_hv[method]))
                        hypervolume_dict[constrain][dataset]["fairness_seed_range"].append(max(help_dict_f[method])-min(help_dict_f[method]))
                        hypervolume_dict[constrain][dataset]["acc_seed_range"].append(max(help_dict_a[method])-min(help_dict_a[method]))

            hv = np.array(hypervolume_dict[constrain][dataset]["hypervolume"]).reshape(-1,1)
            fairness = np.array(hypervolume_dict[constrain][dataset]["fairness"]).reshape(-1,1)
            accurancy = np.array(hypervolume_dict[constrain][dataset]["acc"]).reshape(-1,1)
            fairness_best = np.array(hypervolume_dict[constrain][dataset]["fairness_best"]).reshape(-1,1)
            accurancy_best = np.array(hypervolume_dict[constrain][dataset]["acc_best"]).reshape(-1,1)
            scaled_hv = scaler.fit_transform(hv).tolist()
            scaled_fairness = scaler.fit_transform(fairness).tolist()
            scaled_acc = scaler.fit_transform(accurancy).tolist()
            scaled_fairness_best = scaler.fit_transform(fairness_best).tolist()
            scaled_acc_best = scaler.fit_transform(accurancy_best).tolist()
            
            hypervolume_dict[constrain][dataset]["hypervolume_max_diff"] = max(hypervolume_dict[constrain][dataset]["hypervolume"]) - min(hypervolume_dict[constrain][dataset]["hypervolume"])
            hypervolume_dict[constrain][dataset]["fairness_max_diff"] =  max(hypervolume_dict[constrain][dataset]["fairness"]) - min(hypervolume_dict[constrain][dataset]["fairness"])
            hypervolume_dict[constrain][dataset]["acc_max_diff"] = max(hypervolume_dict[constrain][dataset]["acc"]) - min(hypervolume_dict[constrain][dataset]["acc"])
            hypervolume_dict[constrain][dataset]["hypervolume_scaled_max"] = [value[0] for value in scaled_hv]
            hypervolume_dict[constrain][dataset]["fairness_scaled_max"] = [value[0] for value in scaled_fairness]
            hypervolume_dict[constrain][dataset]["acc_scaled_max"] = [value[0] for value in scaled_acc]
            hypervolume_dict[constrain][dataset]["fairness_best_scaled_max"] = [value[0] for value in scaled_fairness_best]
            hypervolume_dict[constrain][dataset]["acc_best_scaled_max"] = [value[0] for value in scaled_acc_best]
    print(hypervolume_dict)
    with open(file, 'w') as f:
        json.dump(hypervolume_dict, f, indent=4)
def plot_scaled_values(results,result_folder,plot_feature, needed_methods, one_line=False, label=False):
    def rgb(r: int, g: int, b: int) -> str:
        return "#%02x%02x%02x" % (r, g, b)
    c_color = rgb(128, 0, 128)
    c_color_2 = rgb(252, 240, 0)
    c_color_3 = rgb(0,128, 128)
    c_color_4 = rgb(128,128,0)
    styles = {
        "moo": dict(s=15, marker="x", color="red",fullName="Moo",  edgecolors= "black"),
        "cr": dict(s=15, marker="x", color="green",fullName="Correlation remover",edgecolors= "black"),
        "ps_ranker": dict(s=15, marker="x", color="orange",fullName="Sampling",edgecolors= "black"),
        "lfr": dict(s=15, marker="x", color="blue",fullName="LFR",edgecolors= "black"),
        "so": dict(s=15, marker="x", color="black",fullName="Optimizing for accuracy", edgecolors= "black"),
        "moo_ps_ranker": dict(s=15, marker="x", color =c_color_3,fullName="Moo with optionall sampling",edgecolors= "black"),
        "moo+cr": dict(s=15, marker="x", color="purple",fullName="Moo with optionall correlation remover",edgecolors= "black"),
        "moo+sar": dict(s=15, marker="x", color="yellow",fullName="Moo with optionall remove the sensititve attribute",edgecolors= "black"),
        "redlineing": dict(s=15, marker="x", color="grey",fullName="Moo without the senstive attribute",edgecolors= "black"),

        "moo+ps+cr": dict(s=15, marker="o", color ="purple", fullName="Moo with optionall correlation remover xor sampling",edgecolors= "black"),
        "moo+ps*cr": dict(s=15, marker="o", color="red",fullName="Moo with optionall correlation remover and/or sampling",edgecolors= "black"),
        "moo+lfr":  dict(s=15, marker="o", color="blue", fullName="Moo with optioal lfr",edgecolors= "black"),
        "moo+ps+cr+lfr": dict(s=15, marker="o", color="green", fullName="Moo with optional lfr/ps/cr",edgecolors= "black"),
        "moo+ps+lfr": dict(s=15, marker="o", color="orange", fullName="Moo with optional lfr/ps",edgecolors= "black"),
        "ps+cr+lfr": dict(s=15, marker="o", color="grey", fullName="Moo with one preprocessor",edgecolors= "black"),
        "moo+cr+lfr": dict(s=15, marker="o", color="yellow", fullName="Moo with optional lfr or cr",edgecolors= "black"),
    }
    for constrain in results.keys():
    
        datasets = results[constrain].keys()
        plot_width = .6 * len(datasets)
             
        fig, ax = plt.subplots(figsize=(plot_width,2))
        width = 0
        #ss_order = results[constrain].keys()
        max_needed = 0
        for dataset in datasets:
            width += 1
            methods = results[constrain][dataset]["methods"]
            for method in methods:
                idx = results[constrain][dataset]["methods"].index(method)
                data = results[constrain][dataset][plot_feature][idx]
                max_needed = max(max_needed, data)  if method in needed_methods else max_needed
                if method in needed_methods or data > max_needed:
                    color = styles[method]['color']
                    label = styles[method]["fullName"]
                    marker = styles[method]["marker"]
                    if width == 1:
                        ax.scatter(width, data, label=label, color=color, marker=marker)
                    else:
                        ax.scatter(width, data, color=color, marker=marker)
            
            
        

        ax.set_xticks(range(1, len(datasets)+1))
        
        ax.set_xticklabels(datasets, fontsize=12, rotation=90)
       

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
    preprocessors = ["no"]
    preprocessors += ["PreferentialSampling"] if "ps" in method else []
    preprocessors += ["CorrelationRemover"] if "cr" in method else []
    preprocessors += ["LFR"] if "lfr" in method else []
    preprocessors += ["SensitiveAttributeRemover"] if "sar" in method else []
    return preprocessors
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

import numpy as np

def calculate_shapley_value(pareto_set,score):
    # Calculate the shape value using your desired method
    #print(score)
    if len(pareto_set) == 0:
        return 0
    if(score=="hypervolume"):
        hypervolume_obj =  HV(ref_point=np.array([1,1])) 
        shape_value = hypervolume_obj(np.array(pareto_set))
    if(score=="acc"):
        pareto_set = pd.DataFrame(pareto_set)
        shape_value = 1-min(pareto_set[0])
    if(score=="fairness"):
        pareto_set = pd.DataFrame(pareto_set)
        shape_value = 1-min(pareto_set[1])      
    return shape_value
#I think the idea 
# 
def calc_sum(pareto_front):
    hypervolume_obj =  HV(ref_point=np.array([1,1])) 
    vol = hypervolume_obj(np.array(pareto_front))
    pareto_front = pd.DataFrame(pareto_front)
    return sum(pareto_front[0]),sum(pareto_front[1]), vol

def scale_front(pareto_front):
    if len(pareto_front) <=1:
        return 1,1
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(1-pareto_front[0]).reshape(-1,1)), scaler.fit_transform(np.array(1-pareto_front[1]).reshape(-1,1))

def filter_points(pareto_front, pareto_config, preprocessors,scale = False, ref_point = [1,1]):
    if scale:
        pareto_front[0], pareto_front[1] = scale_front(pareto_front)
    score_dict = defaultdict()
    pareto_front = np.array(pareto_front)
    sum_acc, sum_fair, sum_vol = calc_sum(pareto_front)
    #if sum_acc == 0:
    #    print(pareto_front)
    for pre in preprocessors:
        #score_dict[pre] = defaultdict()
        points = []
        for idx,point in enumerate(pareto_front):
            if complex_xor_preproecessing(pareto_config[idx])==pre:
                points.append(point)
        if len(points) == 0:
            points = [ref_point]
        score_dict[pre] = np.array(points)
    return score_dict   
from itertools import permutations
def calculate_shapley_values(data, methods, file, compare = "acc", latex_table = True):
    #shaling should be done in these function
    shapley_values = defaultdict() 
    for constrain in data.keys():
        shapley_values[constrain] = defaultdict() 
        for dataset in data[constrain].keys():
            shapley_values[constrain][dataset] = defaultdict() 
            for seed in data[constrain][dataset].keys():
                for method in methods:
                    if method not in  shapley_values[constrain][dataset].keys():
                        shapley_values[constrain][dataset][method] = defaultdict()
                    pareto_front = data[constrain][dataset][seed][method]["points"]
                    pareto_config = data[constrain][dataset][seed][method]["configs"]
                    preprocessors = get_possible_pre(method)
                    methods_points = filter_points(pareto_front, pareto_config, preprocessors)
                    preprocessors = preprocessors[1:]
                    print(f"{dataset},{constrain},{seed},{method}")
                    div = np.math.factorial(len(preprocessors)) * len(data[constrain][dataset].keys()) * calculate_shapley_value(pareto_front,compare)
                    for perm in permutations(preprocessors):
                        points = np.array(methods_points["no"])
                        for idx,pre in enumerate(perm):
                            new_points = np.vstack((points, methods_points[pre]))  
                            if pre not in shapley_values[constrain][dataset][method].keys():
                                shapley_values[constrain][dataset][method][pre] = defaultdict() 
                                shapley_values[constrain][dataset][method][pre][compare] = 0    
                            shapley_values[constrain][dataset][method][pre][compare] += ((calculate_shapley_value(new_points,compare))- (calculate_shapley_value(points, compare)))/div
                            points = new_points   
                #shapley_values[constrain][dataset][method]["all"]["hypervolumne"] += hypervolume_obj(points) / len(data[constrain][dataset].keys())                                 
        with open(file + "shapley_{}.json".format(compare), 'w') as f:
            json.dump(shapley_values, f, indent=4)
    return  
                   
"""
    post_acc  += methods_points[pre]["acc"]
    post_fair += methods_points[pre]["fairness"]
    post_vol  += methods_points[pre]["hypervolumne"]
    shapley_values[constrain][dataset][method][pre]["acc"] += (post_acc-pre_acc)/div
    shapley_values[constrain][dataset][method][pre]["fairness"]  += (post_fair-pre_fair)/div
    shapley_values[constrain][dataset][method][pre]["hypervolumne"] +=  (post_vol-pre_vol)/div 
"""
def generate_result_table(data):
    
    # Open the LaTeX table format
    for value_key, value_type in {"acc": "acc_best", "fairness": "fairness_best"}.items():
        latex_table = r"""
        \begin{table}[]
        \centering
        \caption{fairness metrics one}
        \label{tab:my-table}
        \begin{tabular}{@{}|l|"""
        
        # Extract unique methods and their count
        datasets = list(set(data))
        num_datasets = len(datasets)
        
        # Add method columns to the table format
        for _ in range(num_datasets):
            latex_table += "l|"
        
        latex_table += r"}\n \toprule \n &"
        
        # Add method names to the table header
        latex_table += " & ".join(datasets) + r" \\ \midrule"
        
        # Iterate through the data and fill in the table
    
        latex_table += f"\n {value_key} &\multicolumn{num_datasets}" +r"{l|}{} \\ \midrule"
        for i, method  in enumerate(data["lawschool"]["methods"]):  
            #lawschool is just a random dataset here because is everytime the same methods  
            latex_table += f"\n {i}"
            for dataset in data.keys():
                val = round(data[dataset][value_type][i], 4)
                latex_table += f"& {val}"
        latex_table += r"\\ \midrule"
        
        # Close the table
        latex_table += r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
        
    return latex_table
def get_method_name(method):
    if method == "redlineing":
        print()
    method_mapping = {
        "moo": "Moo -s",
        "so": "So",
        "cr": "CR -a",
        "ps_ranker": "PS -a",
        "redlineing": "SAR -a",
        "lfr": "LFR -a",
        
        "moo+sar": "SAR -o",
    "moo_ps_ranker": "PS -o",
    "moo+cr": "CR -o",
    "moo+lfr": "LFR -o",

    "moo+sar+cr": "[SAR, CR] -o",
    "moo+sar+ps": "[SAR, PS] -o",
    "moo+sar+ps": "[SAR, PS] -o",
    "moo+ps+cr": "[PS, CR] -o",
    "moo+cr+lfr": "[CR, LFR] -o",
    "moo+ps+lfr": "[PS, LFR] -o",
    "moo_sar_lfr": "[SAR, LFR] -o",

    "moo+ps*cr": "[PS, CR] -m",
    "moo+cr*lfr": "[CR, LFR] -m",
    "moo_sar_ps_com": "[SAR, PS] -m ",

    "moo_sar_cr_lfr": "[SAR, CR, LFR] -o",
    "moo_sar_ps_lfr": "[SAR, PS, LFR] -o",
    "moo+sar+cr+ps": "[SAR, CR, PS] -o",
    "moo+ps+cr+lfr": "[PS, CR, LFR] -o",
    "moo+sar+cr+ps": "[SAR, CR, PS] -o",

    "moo_sar_ps_cr_lfr": "[all] -o",
    "sar_cr_ps_lfr": "[all] -of",
    "cp": "CP",
    "bp": "BP",
    }
    return method_mapping[method]

method_groups = [[range(2,5)], [range(6,9)],[range(10,28)]]


def test_table(alldata, file = "/home/till/Desktop/redlineing/table_cp.txt"):
    for constrain, data in alldata.items():
        latex_table = ""
        
        latex_table += fr"""
        \begin{{table}}
        \caption{{Results of \text{constrain}}}
        \begin{{center}}"""
        latex_table += r"\scalebox{0.55}{" +"\n"
        latex_table += fr"""\centering
        \label{{tab:{constrain}}}
        \begin{{tabular}}{{l cccc c cccc c cccc}}
        \hline
        Method & \multicolumn{{{len(data)}}}{{c}}{{Accuracy}} & & \multicolumn{{{len(data)}}}{{c}}{{Fairness}}  & & \multicolumn{{{len(data)}}}{{c}}{{Hypervolume}}\\
        \cline{{2-5}} \cline{{7-10}} \cline{{12-15}}
        & {' & '.join(data.keys())} & & {' & '.join(data.keys())} & & {' & '.join(data.keys())} \\
            """    
        method_count = len(data[list(data.keys())[0]]["methods"])  # Assuming the count is consistent across datasets    
        for i in range(method_count):
            method_name = get_method_name(data[list(data.keys())[0]]["methods"][i])
            latex_table +=  "\n"+ r"\text{" + f"{method_name}" + r"}"
            for value_type in ["acc_best", "fairness_best", "hypervolume"]:
                for dataset in data.keys():
                    val = round(data[dataset][value_type][i], 4)
                    if data[dataset][value_type][i] == max(data[dataset][value_type]):
                        latex_table += r" & \textbf{" + str(val) + r"}"
                    else:
                        latex_table += f" & {val}"
                    latex_table += r" & " if dataset == "compass"  and value_type != "hypervolume" else r""
            latex_table += r" \\"
            
        latex_table += r"""
        \bottomrule
        \end{tabular}
        }
        \end{center}
        \end{table}
        """
        latex_table += "\n"
        #print(latex_table)
        with open(file, "a+") as f:
            f.write(latex_table)
    
def check_if_bold(value_type, val, method_name):
    if method_name == "[all] -of":
        return False
    if value_type == "acc":
        return val >= 0.01
    else:
        return val >= 0.05

from mergedeep import merge
def generate_latex_table(file = "/home/till/Desktop/shapley_values/table_marked.txt"):
    data = []
    for post in ["acc", "fairness", "hypervolume"]:
        with open("/home/till/Desktop/shapley_values/shapley_{}.json".format(post), 'r') as f:
            data.append(json.load(f))
    data = merge(data[0], data[1], data[2])
    for constrain, data in data.items():
        latex_table = ""
        latex_table += fr"""
        \begin{{table}}
            
        \begin{{center}}"""
        latex_table += r"\scalebox{0.55}{" +"\n"
        latex_table += fr"""\centering
        
        \begin{{tabular}}{{l cccc c cccc c cccc}}
        \hline
        Method & \multicolumn{{{len(data)}}}{{c}}{{Accuracy Gain}} & & \multicolumn{{{len(data)}}}{{c}}{{Fairness Gain}}  & & \multicolumn{{{len(data)}}}{{c}}{{Hypervolume Gain}}\\
        \cline{{2-5}} \cline{{7-10}} \cline{{12-15}}
        & {' & '.join(data.keys())} & & {' & '.join(data.keys())} & & {' & '.join(data.keys())} \\
                """    
        #method_count = len(data["lawschool"].keys())  # Assuming the count is consistent across datasets    
        for method in data["lawschool"].keys():
            method_name = get_method_name(method)
            latex_table += f" \cline{{2-5}} \cline{{7-10}} \cline{{12-15}}"
            latex_table +=  "\n"+ r"\textbf{" + f"{method_name}" + r"} \\"
            for prepro in data["lawschool"][method].keys():
                latex_table +=  "\n"+ r"\text{" + f"{prepro}" + r"}"
                for value_type in data["lawschool"][method][prepro].keys(): 
                    for dataset in data.keys():
                        val = data[dataset][method][prepro][value_type] # that has to change to something that makes sense
                        if val > 0.01 or val == 0:
                            if check_if_bold(value_type,val, method_name):
                                latex_table += r" & \textbf{" + f"{val: .2f}" + r"}"
                            else:
                                latex_table += f" & {val: .2f}"
                        else:
                            latex_table += f"& {val: .2e}"
                        latex_table += r" & " if dataset == "compass"  and value_type != "hypervolume" else r""
                latex_table += r" \\"
                
        latex_table += r"""
        \bottomrule
        \end{tabular}
        }
        }"""

        latex_table += f"""\caption{{Shapley Values for {constrain}. For combination without \gls{{mooE}}, the base values is zero. 
        For combination with \gls{{mooE}}, the base value is the \gls{{mooE}} value. 
        Which leads to lot higher values for the combinations with \gls{{mooE}}.}}"""
        latex_table += r"""
        \label{tab:sv}
        \end{center}
        \end{table}
        """
        latex_table += "\n"
        #print(latex_table)
        with open(file, "a+") as f:
            f.write(latex_table)

def get_2_important(data, excluded_keys):
    max_importance_of = float("-inf")
    highest_importance_name_of = None
    max_importance = float("-inf")
    highest_importance_name = None
    
    for key, value in data.items():
        if key not in excluded_keys:
            individual_importance = value.get("individual importance", 0)
            if individual_importance > max_importance_of:
                max_importance_of= individual_importance
                highest_importance_name_of = key
        individual_importance = value.get("individual importance", 0)
        if individual_importance > max_importance:
            max_importance= individual_importance
            highest_importance_name = key    

    return highest_importance_name_of, highest_importance_name


def generate_fanvoa_table(methods, file = "/home/till/Desktop/fanova/table_marked.txt"):
     
    for constrain in ["consistency_score", "equalized_odds", "demographic_parity"]:
        latex_table = ""
        latex_table += fr"""
        \begin{{table}}
            
        \begin{{center}}"""
        latex_table += r"\scalebox{0.43}{" +"\n"
        latex_table += fr"""\centering
        
        \begin{{tabular}}{{l cccc c cccc}}
        \hline
        Method & \multicolumn{{4}}{{c}}{{Accuracy fANOVA}} & & \multicolumn{{4}}{{c}}{{Fairness fANOVA}} \\
        \cline{{2-5}} \cline{{7-10}}
         & lawschool & german & adult & compass & & lawschool & german & adult & compass \\
                """    
        ov_hi_of = []
        #method_count = len(data["lawschool"].keys())  # Assuming the count is consistent across datasets    
        for method in methods:
            method_name = get_method_name(method)
            latex_table += f" \cline{{2-5}} \cline{{7-10}}"
            latex_table +=  "\n"+ r"\textbf{" + f"{method_name}" + r"} \\"
            

            name = [ "fair_preprocessor", "Best other hyperparameter", "Best other hyperparameter"]       
            for i in range(0,3):
                latex_table +=  "\n"+ r"\text{" + f"{name[i]}" + r"}"
                for value_type in ["performance", "fairness"]:
                    for dataset in ["lawschool","german", "adult","compass"]:
                        with open("/home/till/Desktop/fanova/{}/{}_{}.json".format(value_type, value_type, method), 'r') as f:
                            data = json.load(f) 
                        if "-m" in method_name:          
                            prepro_types = ["data_preprocessor:__choice__", "feature_preprocessor:__choice__"]
                        elif method == "moo+cr":
                            prepro_types = ["feature_preprocessor:__choice__"]
                        else:
                            prepro_types = ["fair_preprocessor:__choice__"]
                        
                        try:
                            fanova_values = data[dataset][constrain]
                        except: 
                            print(f"{method}/{dataset}/{constrain} not found")
                            continue
                        hip_of, hip = get_2_important(fanova_values, prepro_types)
                        ov_hi_of.append(hip_of)
                        prepro_types.append(hip_of)
                        #print(len(prepro_types))
                        if i > len(prepro_types)-1:
                            continue
                        #print(i)
                        val = fanova_values[prepro_types[i]]["individual importance"] # that has to change to something that makes sense
                        if val > 0.01 or val == 0:
                            val_t = f"{val: .2f}"
                        else:
                            val_t = f"{val: .2e}"
                        if prepro_types[i] == hip:
                            latex_table += r" & \textbf{" + val_t + r"}"
                        else:
                            latex_table +=r"& "+ val_t
                        latex_table += r" & " if dataset == "compass"  and value_type != "fairness" else r""
                latex_table += r" \\" + "\n"
                        
        latex_table += r"""
        \bottomrule
        \end{tabular}
        }
        """

        latex_table += f"""\caption{{FANOVA Values for {constrain}.}}"""
        latex_table += r"""
            \label{tab:fanova}
            \end{center}
            \end{table}
            """
        latex_table += "\n"
        #print(latex_table)
        with open("/home/till/Desktop/fanova/table_marked.txt", "a+") as f:
            f.write(latex_table)
        from collections import Counter
        print(dict(Counter(ov_hi_of)))
# Create an empty list to store processed data

def get_used_methods(data):
    methods = []
    for dataset in data.keys():
        for method in data[dataset].keys():
            if method not in methods:
                methods.append(method)
    return methods


def create_pareto_dominace_table(file= "/home/till/Desktop/domination.json"):
    # we do not speak about time complexity here.
    with open(file, 'r') as f:
            data = json.load(f)
    
    latex_table = ""
    latex_table += fr"""
    \begin{{table}}
            
    \begin{{center}}""" + "\n"
    latex_table += r"\scalebox{0.55}{" +"\n"
    latex_table += fr"""\centering
        
    \begin{{tabular}}{{l cccc}}
    \hline
    Method & {' & '.join(data.keys())} & & {' & '.join(data.keys())} & & {' & '.join(data.keys())} \\
                """    
    #method_count = len(data["lawschool"].keys())  # Assuming the count is consistent across datasets    
    for constrain, data in data.items():
        latex_table += "\n" + f" \cline{{}}"
        latex_table +=  "\n"+ r"\textbf{" + f"{constrain}" + r"} \\"
        methods = get_used_methods(data)
        for method in methods:
                method_name = get_method_name(method)
                latex_table +=  "\n"+ r"\text{" + f"{method_name}" + r"} "
                for data_name in ["lawschool", "german", "adult", "compass"]:
                    if data.get(data_name) is not None and method in data[data_name].keys():
                        val = data[data_name][method] # that has to change to something that makes sense
                        if val >= 5:
                            latex_table +=  r"& \textbf{" + f"{val}" + r"}"
                        else:
                            latex_table += f"& {val}"
                    else:
                        latex_table += f" & 0"
                latex_table += r" \\"
                    
    latex_table += r"""
    \bottomrule
    \end{tabular}
    }
    }"""

    latex_table += f"""\caption{{number of seeds \gls{{mooE}} is dominated.}}"""
    latex_table += r"""
    \label{tab:pd}
    \end{center}
    \end{table}
        """
    with open("/home/till/Desktop/dom_table.txt", "a+") as f:
        f.write(latex_table) 

from collections import OrderedDict
method_mapping = OrderedDict([
    ("moo", {"name": "Moo -a", "type": "baseline"}),
    ("so", {"name": "So", "type": "baseline"}),
    ("cr", {"name": "CR -a", "type": "baseline"}),
    ("ps_ranker", {"name": "PS -a", "type": "baseline"}),
    ("redlineing", {"name": "SAR -a", "type": "baseline"}),
    ("lfr", {"name": "LFR -a", "type": "baseline"}),
    ("moo+sar", {"name": "SAR -o", "type": "optional"}),
    ("moo_ps_ranker", {"name": "PS -o", "type": "optional"}),
    ("moo+cr", {"name": "CR -o", "type": "optional"}),
    ("moo+lfr", {"name": "LFR -o", "type": "optional"}),
    ("moo+sar+cr", {"name": "[SAR, CR] -o", "type": "combination"}),
    ("moo+sar+ps", {"name": "[SAR, PS] -o", "type": "combination"}),
    ("moo+sar+ps", {"name": "[SAR, PS] -o", "type": "combination"}),
    ("moo+ps+cr", {"name": "[PS, CR] -o", "type": "combination"}),
    ("moo+cr+lfr", {"name": "[CR, LFR] -o", "type": "combination"}),
    ("moo+ps+lfr", {"name": "[PS, LFR] -o", "type": "combination"}),
    ("moo_sar_lfr", {"name": "[SAR, LFR] -o", "type": "combination"}),
    ("moo+ps*cr", {"name": "[PS, CR] -m", "type": "combination"}),
    ("moo+cr*lfr", {"name": "[CR, LFR] -m", "type": "combination"}),
    ("moo_sar_ps_com", {"name": "[SAR, PS] -m", "type": "combination"}),
    ("moo_sar_cr_lfr", {"name": "[SAR, CR, LFR] -o", "type": "combination"}),
    ("moo_sar_ps_lfr", {"name": "[SAR, PS, LFR] -o", "type": "combination"}),
    ("moo+sar+cr+ps", {"name": "[SAR, CR, PS] -o", "type": "combination"}),
    ("moo+ps+cr+lfr", {"name": "[PS, CR, LFR] -o", "type": "combination"}),
    ("moo+sar+cr+ps", {"name": "[SAR, CR, PS] -o", "type": "combination"}),
    ("moo_sar_ps_cr_lfr", {"name": "[all] -o", "type": "combination"}),
    ("sar_cr_ps_lfr", {"name": "[all] -of", "type": "combination"}),
        
])    
    
   
colors = [
    "#008000", "#0000FF", "#FFFF00", "#FFA500", 
    "#800080", "#FFC0CB", "#008080", "#A52A2A", "#00FFFF", 
    "#FF00FF", "#00FF00", "#4B0082", "#E6E6FA", "#800000", 
    "#808000", "#000080", "#FFDAB9", "#40E0D0", "#FFD700", 
    "#C0C0C0", "#EE82EE", "#00FFFF", "#FF6F61", "#DC143C", 
    "#2E8B57", "#FF00FF", "#FFFFF0", "#E0115F", "#007FFF",
    "#F5F5DC", "#7FFF00", "#F0FFF0", "#F5F5F5", "#F0FFFF",
    "#FDF5E6", "#FFDEAD", "#FFDAB9", "#F0F8FF", "#F5FFFA",
]



def barplot_results(data, comparison, shortName, save_folder):
    dataset_order = ['german',"compass", "lawschool", "adult"] 
    method_order = [details['name'] for method, details in method_mapping.items()]
    # TODO: think about vissaualize the different methods:
    # baselines optically awy from the rest
    # everytime 
    # optional
    # rest compinations
    # mark the best of every group and the best overall
    #create an empty list to store processed data  
    all_dfs = []

    # Process data for each metric
    for metric, metric_data in data.items():
        # Process sub-metrics within each metric
        for dataset, dataset_data in metric_data.items():
            # Create a list of dictionaries for the DataFrame
            df_data = []
            for i, method in enumerate(dataset_data['methods']):
                # rename stranged named methods
                method = method_mapping[method]["name"]
                #method_type = method_mapping[method]["type"]
                df_data.append({
                    'dataset': dataset,
                    'method': method,
                    'metric': metric,
                    'value': dataset_data[comparison][i],
                    #'type': method_type
                })

            # Create a DataFrame from the list of dictionaries
            sub_metric_df = pd.DataFrame(df_data)

            # Append the processed DataFrame to the list
            all_dfs.append(sub_metric_df)

    # Concatenate all DataFrames
    final_df = pd.concat(all_dfs)
    # Set up Seaborn aesthetics
    

    # Create separate bar plots for each metric
    
    
    for metric in final_df['metric'].unique():
        sns.set(style="whitegrid")
        plt.figure(figsize=(20, 10))  #6, 7.8
        metric_df = final_df[final_df['metric'] == metric]
        metric_df['method'] = pd.Categorical(metric_df['method'], categories=method_order, ordered=True)
        metric_df['dataset'] = pd.Categorical(metric_df['dataset'], categories=dataset_order, ordered=True)
        metric_df = metric_df.sort_values(['dataset','method']).reset_index(drop=True)
        #sns.set_palette(colors)
        ax = sns.barplot(data=metric_df, y="dataset",x="value", hue="method", orient="h", 
                         hue_order=method_order,palette=sns.color_palette(colors, len(colors)),) # palette=sns.color_palette(colors, len(colors)),
        y_min = sub_metric_df['value'].min()
        x_max =  sub_metric_df['value'].max()
        ax.set_xlim(0, 1.1)
         # Add vertical lines to mark specific methods for each dataset
        #split_lines = [-0.24, 0.08]
        best_indicies = list(enumerate(metric_df.groupby('dataset')['value'].idxmax()))
        new_index = [(x%len(metric_df["method"].unique()))*len(metric_df["dataset"].unique()) + i for i,x in best_indicies]
        
        for i,p in enumerate(ax.patches):
            text_color = "black"
            #print("%.4f" % p.get_width())
            #if i in [4,5,6,720,21,22,23, 36,37,38]: #,7,
            #    ax.axhline(y=p.get_y()+p.get_height(), color='gray', linestyle='--')
            if i in new_index:
                text_color = "red"
            ax.annotate("%.3f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha="left", va="center", color=text_color)
            
               
        #for index, row in best_methods_dataset.iterrows():
        #    
        plt.title(f"Comparison of {metric} by Dataset and Method")
        plt.ylabel("Dataset")
        plt.xlabel(f"{shortName}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position legend outside the plot
        #plt.legend([],[], frameon=False)
        plt.tight_layout()  # Ensure proper layout
        #plt.show()
        #break
        plt.savefig("{}{}/{}_pv_comparison.png".format(save_folder, comparison, metric))  # Save the plot as an image
        plt.close()  # Close the current plot

def check_pareto_front(data):
    domination = defaultdict()
    for constrain, constrain_data in data.items():
        domination[constrain] = defaultdict()
        # Process sub-metrics within each metric
        for dataset, dataset_data in constrain_data.items():
            domination[constrain][dataset] = defaultdict()
            for seed, seed_data in dataset_data.items():
                moo_front = seed_data["moo"]["pareto_set"]
                #dataset_data["moo"]["pareto_config"] = 
                moo_config = [{'method': 'moo', **item} for item in  seed_data["moo"]["pareto_config"]]
                # Create a list of dictionaries for the DataFrame
                for method, method_data in seed_data.items():
                    if method == "moo" or method == "so":
                        continue
                    new_front = defaultdict()
                    new_front["points"] = pd.concat([moo_front, method_data["pareto_set"]], ignore_index= True)
                    new_front["configs"] = moo_config + [{'method': method, **item} for item in  method_data["pareto_config"]]
                    pareto_points, pareto_config = pareto_set(new_front)
                    #get new pareto front and check if only method is the pareto front
                    #dominates = if pareto_config
                    if all(item.get('method') == method for item in pareto_config):
                        
                        if  method not in domination[constrain][dataset].keys():
                            domination[constrain][dataset][method] = 1
                        else:
                            domination[constrain][dataset][method] += 1
    #write domination to file
    with open("/home/till/Desktop/domination.json", 'w') as f:  
        json.dump(domination, f, indent=4)
    return

if __name__ == "__main__":
    #methods = ["moo_ps_ranker","moo","moo+cr", "ps_ranker"]
    #"moo", "so", #baslines 
    #            "ps_ranker", "cr", #one prepreproessor
    #            "moo_ps_ranker","moo+cr", "moo+lfr", #one prepreproessor
    methods = [ 
        #"moo",
        #"so",
        #preprocessor every time
        #"redlineing",
        #"cr",
        #"ps_ranker",
        #"lfr",
        #preprocessor optional
        "moo+sar",     
        "moo_ps_ranker",
        "moo+cr",
        "moo+lfr",
        #multiple preprocessor 
        #two:
        "moo+sar+cr",
        "moo+sar+ps", 
        "moo+ps+cr",
        "moo+cr+lfr",
        "moo+ps+lfr",
        "moo_sar_lfr",
        "moo+sar+ps",


        #and/or combinations:
        "moo+ps*cr",
        "moo+cr*lfr",
        "moo_sar_ps_com",

        #three:
        "moo_sar_cr_lfr",
        "moo_sar_ps_lfr",
        "moo+sar+cr+ps",
        "moo+ps+cr+lfr",
        "moo+sar+cr+ps",

        #all:
        "moo_sar_ps_cr_lfr",
        "sar_cr_ps_lfr"
        #"cp",
        #"bp"
          
        ]
  
    #make_plot_3(data)
    #data = load_data_particully("/home/till/Desktop/psoss_val/", "200timesstrat")
    #datasets = ["german","adult", "compass","lawschool"],
    #constrains = ["consistency_score"],
    #folders=["one_rf_seed_1", "one_rf_seed"],
    #"12345","25","42","45451", "97","13","27","39","41","53"
    #seeds= ["12345","25","42","45451", "97","13","27","39","41","53"]
    #make_difference_plot(data,"best")
    #methods = ["moo","cr"]
    #data = load_data("/home/till/Desktop/cross_val/", "200timesstrat", methods)
    #methods = ["moo","ps_ranker","moo_ps_ranker"]
    #print()
    #for compare in ["hypervolume"]:
    #    calculate_shapley_values(data,methods,file="/home/till/Desktop/shapley_values/", compare=compare)
    #print(sv)
    #make_plot_3(data)
    #deep_dive = defaultdict()
    #for constrain in data.keys():
    #    deep_dive[constrain] = defaultdict()
    #    for dataset in data[constrain].keys():
    #        deep_dive[constrain][dataset] = defaultdict()
    #for method in methods:
    #            deep_dive[method] = defaultdict()
    #            for seed in seeds:
    #                deep_dive[method][seed] = data["error_rate_difference"]["adult"][seed][method]["pareto_config"]
    #generate_latex_table()                
    generate_fanvoa_table(methods)   
    #create_pareto_dominace_table()
    file = "/home/till/Documents/auto-sklearn/tmp/scaled_results_all_cp.json"
    #calc_hypervolume(data, file)
    with open(file) as f:
        results = json.load(f)
    #check_pareto_front(data)
    #test_table(results)
    #names = ["hypervolume[scaled]","accurancy[bestScaled]", "fairness[bestScaled]",
    #                                "accurancy[avgScaled]", "fairness[avgScaled]",
    #                               "accurancy[best]", "fairness[best]",
    #                                 "accurancy[avg]", "fairness[avg]", "hypervolume"]
    #names = ["test"]
    #comparisons = ["hypervolume_scaled_max","acc_best_scaled_max", "fairness_best_scaled_max", 
    #                               "acc_scaled_max", "fairness_scaled_max",
    #                               "acc_best", "fairness_best",
    #                                "acc", "fairness", "hypervolume"]
    #for i, comparison in enumerate(comparisons):
    #barplot_results(results, comparisons[0], names[0],"/home/till/Desktop/redlineing/all/")
       
    #plot_scaled_values(results,"/home/till/Desktop/redlineing/all/","hypervolume_scaled_max", methods)


    #["hypervolume_scaled_max","acc_best_scaled_max", "fairness_best_scaled_max", 
    #                                "acc_scaled_max", "fairness_scaled_max",
    #                               "acc_best", "fairness_best",
    #                              "acc", "fairness"]