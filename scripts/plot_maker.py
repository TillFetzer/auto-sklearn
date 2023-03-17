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

def make_plot(methods=["moo","cr"], dataset= "adult", runtime=10800):
    sns.set_context("paper", font_scale=0.6)
    figsize = (20, 8)
    dpi = 300
    main_size = 20
    plot_offset = 0.05
    title_size = 18
    label_size = 16
    tick_size = 12
    with open('/home/till/Documents/auto-sklearn/results_t.json') as f:
        data = json.load(f)
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
        "cr_points": dict(s=15, marker="o", edgecolors="black", facecolors="none"),
        "cr_pareto": dict(s=4, marker="o", color="black", linestyle="-", linewidth=2, line ="dotted"),
        "cr_test_points": dict(s=15, marker="o", edgecolors="black", facecolors="none", color ="black"),
        "moo_test_points": dict(s=15, marker="o", edgecolors=c_chocolate,color=c_chocolate, facecolors="none"),
    }


    ax = val_ax

    # Highlight val pareto front and how things moved
    
    plot(result_0_val, ax=ax, **styles["moo_points"])
    pareto_plot(result_0_val, ax=ax, **styles["moo_pareto"])
    # Show the test pareto but faded
    plot(result_1_val, ax=ax, **styles["cr_points"])
    pareto_plot(result_1_val, ax=ax, **styles["cr_pareto"])
    # test_scores.plot( ax=ax, alpha=alpha, **styles["test_points"])

    ax = test_ax

    # Highlight val pareto front and how things moved
    
    plot(result_0_test, ax=ax, **styles["moo_test_points"])
   
    # Show the test pareto but faded
    plot(result_1_test, ax=ax, **styles["cr_test_points"])
   
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
    all_costs = np.array(all_costs)
    sort_by_first_metric = np.argsort(all_costs[:, 0])
    efficient_points = pareto_front(all_costs, is_loss=True)
    pareto_set = []
    for argsort_idx in sort_by_first_metric:
            if not efficient_points[argsort_idx]:
                continue
            pareto_set.append(all_costs[argsort_idx,:])
    pareto_set = pd.DataFrame(pareto_set)
    if len(pareto_set.index)<1:
        pareto_set.loc[-1] = [-1, pareto_set[1][0]]
        pareto_set.index = pareto_set.index + 1  # shifting index
        pareto_set = pareto_set.sort_index()
        pareto_set =  pareto_set.append([[2, pareto_set[1][0]]])
    return pareto_set



def load_data(filepath, runetime):
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
                for method in os.listdir(seed_path):
                    data[constrain][dataset][seed][method] = defaultdict()
                    data[constrain][dataset][seed][method]["points"] = []
                    data[constrain][dataset][seed][method]["pareto_front"] = []
                    file  = "{}/{}/{}/runhistory.json".format(seed_path,method,runetime)
                    with open(file) as f:
                        ds = json.load(f)
                    for d in ds["data"]:
                        try:
                            if d[1][2]["__enum__"] != "StatusType.SUCCESS":
                                continue
                            point = d[1][0]
                    

                        #if run was not sucessfull no train loss is generated
                        #these happened also for sucessfull runs for example timeout
                        except KeyError:
                            continue 
                        data[constrain][dataset][seed][method]['points'].append(point)
                    data[constrain][dataset][seed][method]['points'] = pd.DataFrame(data[constrain][dataset][seed][method]['points'])
                    data[constrain][dataset][seed][method]['pareto_front']  = pareto_set(data[constrain][dataset][seed][method]["points"])
                    #print("file:{},pareto_set:{}".format(file, data[constrain][dataset][seed][method]['pareto_front']))
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
        nrows=len(data),
        ncols=len(data[list(data.keys())[0]]), #needs to be more flexible for nowe is ok
        sharey=True,
        figsize=figsize,
    )
    fig.supxlabel("error",fontsize=label_size)
    fig.supylabel("equalized_odds", fontsize=label_size)
    
    alpha = 0.1

    styles = {
        "moo_points": dict(s=15, marker="o", color="red"),
        "moo_pareto": dict(s=4, marker="o", color="red", linestyle="-", linewidth=2),
        "cr_points": dict(s=15, marker="o", color="blue"),
        "cr_pareto": dict(s=4, marker="o", color="blue", linestyle="-", linewidth=2),
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
                cr_points = data[constrain][dataset][seed]['cr']['points']
                redlineing_points = data[constrain][dataset][seed]['redlineing']['points']
                moo_pf = data[constrain][dataset][seed]['moo']['pareto_front']
                cr_pf = data[constrain][dataset][seed]['cr']['pareto_front']
                redlineing_pf = data[constrain][dataset][seed]['redlineing']['pareto_front']
                plot(moo_points, ax=ax, **styles["moo_points"], alpha = alpha)
                #eaf_plot = EmpiricalAttainmentFuncPlot()
                pareto_plot(moo_pf, ax=ax, **styles["moo_pareto"])
                plot(cr_points, ax=ax, **styles["cr_points"], alpha = alpha)
                pareto_plot(cr_pf, ax=ax, **styles["cr_pareto"])
                plot(redlineing_points, ax=ax, **styles["redlineing_points"], alpha = alpha)
                pareto_plot(redlineing_pf, ax=ax, **styles["redlineing_pareto"])
                if len(moo_pf.index)>3:
                    moo_pf.drop(index=moo_pf.index[[0,-1]],inplace=True)
                if len(cr_pf.index)>3:
                    cr_pf.drop(index=cr_pf.index[[0,-1]],inplace=True)
                if len(redlineing_pf.index)>3:
                    redlineing_pf.drop(index=redlineing_pf.index[[0,-1]],inplace=True)
                local_min_x = min(min(moo_pf[0]), min(cr_pf[0]), min(redlineing_pf[0]))
                local_min_y = min(min(moo_pf[1]), min(cr_pf[1]), min(redlineing_pf[1]))
                local_max_x = max(max(moo_pf[0]), max(cr_pf[0]), max(redlineing_pf[0]))
                local_max_y = max(max(moo_pf[1]), max(cr_pf[1]), max(redlineing_pf[1]))
                #local_min_x = min(min(moo_points[0]), min(cr_points[0]), min(redlineing_points[0]))
                #local_min_y = min(min(moo_points[1]), min(cr_points[1]), min(redlineing_points[1]))
                #local_max_x = max(max(moo_points[0]), max(cr_points[0]), max(redlineing_points[0]))
                #local_max_y = max(max(moo_points[1]), max(cr_points[1]), max(redlineing_points[1]))
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



def make_plot_3(data):
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
        nrows=len(data),
        ncols=len(data[list(data.keys())[0]]), #needs to be more flexible for nowe is ok
        sharey=True,
        figsize=figsize,
    )
    fig.supxlabel("error",fontsize=label_size)
    fig.supylabel("equalized_odds", fontsize=label_size)
    
    alpha = 0.1

    styles = {
        "moo_points": dict(s=15, marker="o", color="red"),
        "moo_pareto": dict(s=4, marker="o", color="red", linestyle="-", linewidth=2),
        "cr_points": dict(s=15, marker="o", color="blue"),
        "cr_pareto": dict(s=4, marker="o", color="blue", linestyle="-", linewidth=2),
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
            moo_pf = []
            #cr_points = []

            for seed in data[constrain][dataset].keys():
                moo_pf.append(np.array(data[constrain][dataset][seed]['moo']['pareto_front']))
                #cr_points.append()
                #seed = "25" 
            #moo_points = data[constrain][dataset][seed]['moo']['points']
            #cr_points = data[constrain][dataset][seed]['cr']['points']
            #redlineing_points = data[constrain][dataset][seed]['redlineing']['points']
            #moo_pf = data[constrain][dataset][seed]['moo']['pareto_front']
            #cr_pf = data[constrain][dataset][seed]['cr']['pareto_front']
            #redlineing_pf = data[constrain][dataset][seed]['redlineing']['pareto_front']
            #TODO: transform in shape(seed, point, metric)
            #TODO: also need to thing about that not every seed has the same amount of points
            #moo_pf = np.vstack(moo_pf)
            levels = [1, len(moo_pf) // 2, len(moo_pf)]
            surfs = get_empirical_attainment_surface(costs=moo_pf, levels= levels)
            eaf_plot = EmpiricalAttainmentFuncPlot()
            eaf_plot.plot_multiple_surface(
                ax,
                surfs=surfs,
                colors=["red","red","red"], 
                linestyles=["-","dotted","dashed"],
                )

              



if __name__ == "__main__":
    data = load_data("/home/till/Documents/auto-sklearn/tmp/", "200times")
    make_plot_3(data)