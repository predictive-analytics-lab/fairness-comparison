"""Generate figures"""

import subprocess
from pathlib import Path
import fire
import pandas as pd

from matplotlib import pyplot as plt

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import TAGS

# The graphs to generate: (xaxis measure, yaxis measure)
GRAPHS = [('DIbinary', 'accuracy'), ('sex-TPR', 'accuracy'),
          ('race-TPR', 'accuracy'), ('age-TPR', 'accuracy')]


def run(rfigures=True, dataset=get_dataset_names(), graphs=GRAPHS):
    for dataset_obj in DATASETS:
        if dataset_obj.get_dataset_name() not in dataset:
            continue

        print("\nGenerating graphs for dataset:" + dataset_obj.get_dataset_name())
        for sensitive in dataset_obj.get_sensitive_attributes():
            for tag in TAGS:
                print("    type:" + tag)
                filename = dataset_obj.get_results_filename(sensitive, tag)
                make_all_graphs(filename, graphs)
    if rfigures:
        print("Generating additional figures in R...")
        generate_rmd_output()


def make_all_graphs(filename, graphs):
    try:
        f = pd.read_csv(filename)
    except:
        print("File not found:" + filename)
        return
    else:
        o = Path(filename).parts[-1].split('.')[0]

        if graphs == 'all':
            graphs = all_possible_graphs(f)

        for xaxis, yaxis in graphs:
            generate_graph(f, xaxis, yaxis, o)


def all_possible_graphs(f):
    graphs = []
    measures = list(f.columns.values)[2:]
    for m1 in measures:
        for m2 in measures:
            graphs.append((m1, m2))
    return graphs


def generate_graph(f, xaxis_measure, yaxis_measure, title):
    try:
        col1 = f[xaxis_measure]
        col2 = f[yaxis_measure]
    except:
        print("Skipping measures: " + xaxis_measure + " " + yaxis_measure)
        return
    else:

        if len(col1) == 0:
            print("Skipping graph containing no data:" + title)
            return

        if col1[0] == 'None':
            print("Skipping missing column %s" % xaxis_measure)
            return

        if col2[0] == 'None':
            print("Skipping missing column %s" % yaxis_measure)
            return

        save_path = Path("results") / Path("analysis") / Path(title)
        save_path.mkdir(parents=True, exist_ok=True)
        # scale = scale_color_brewer(type='qual', palette=1)
        # d3.schemeCategory20
        # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
        #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
        #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
        colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                  "#17becf", "#9edae5"]

        plt.style.use('ggplot')  # optional
        fig, plot = plt.subplots(figsize=(20, 6))
        for i, (name, group) in enumerate(f.groupby('algorithm')):
            plot.plot(group[xaxis_measure], group[yaxis_measure], 'o', label=name, c=colors[i])
        plot.set_xlabel(xaxis_measure)
        plot.set_ylabel(yaxis_measure)
        plot.set_title(title)
        lgd = plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
        figure_path = str(save_path / Path(f"{xaxis_measure}-{yaxis_measure}.png"))
        fig.savefig(figure_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        print(xaxis_measure, yaxis_measure)


def generate_rmd_output():
    subprocess.run(["Rscript", "results/generate-report.R"])


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
