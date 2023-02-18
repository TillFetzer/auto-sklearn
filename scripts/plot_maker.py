import json
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys


def plot(data,ax,kwargs,):
        if ax is None:
            ax = plt.gca()
        if "marker" not in kwargs:
            kwargs["marker"] = "x"
        x, y = (data[0], data[1])
        ax.scatter(x, y, **kwargs)
        return ax 
def pareto_plot(data,ax,**kwargs,):
      
    plot(ax=ax, **kwargs)

      
    if ax is None:
                ax = plt.gca()

    if "marker" not in kwargs:
        kwargs["marker"] = "o"
        xs,ys = data[0], data[1]
        new_xs = list(xs)
        new_ys = list(ys)
        ax.step(new_xs, new_ys, where="post", **kwargs)

        return ax

def make_plot(methods=["moo","crr"], dataset= "adult", runtime=10800):
    figsize = (20, 8)
    dpi = 300
    main_size = 20
    plot_offset = 0.05
    title_size = 18
    label_size = 16
    tick_size = 12
    with open('/home/till/Documents/auto-sklearn/results_t.json') as f:
        data = json.load(f)
    result_0 = data[(data.dataset == "adult") & (dataset.methods==methods[0])&(dataset.runtime == runtime)]
    result_1 = data[(data.dataset == "adult") & (dataset.methods==methods[0])&(dataset.runtime == runtime)]
    result_0_val = pd.DataFrame(result_0["val"])
    result_0_test = pd.DataFrame(result_0["test"])
    result_1_val = pd.DataFrame(result_1["val"])
    result_1_test = pd.DataFrame(result_1["test"])
    fig, (val_ax, test_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
    )
    fig.supxlabel(methods[0],fontsize=label_size)
    fig.supylabel(methods[1].replace("_", " ").capitalize(), fontsize=label_size)

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
        "cr_pareto": dict(s=4, marker="o", color="black", linestyle="-", linewidth=2),
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
                   Line2D([0], [0], color="black", lw=4, label='moo without preprocessing')]
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.legend(legend_elements, loc=3,  prop={'size': 6})
    plt.savefig(f"./figures/experiment_1_{dataset}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)


if __name__ == "__main__":
    make_plot()