import fire
import pandas as pd
from pathlib import Path
import subprocess

from ggplot import scale_color_manual, ggplot, aes, geom_point, ggtitle
from matplotlib import pyplot as plt

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import TAGS

# The graphs to generate: (xaxis measure, yaxis measure)
GRAPHS = [('DIbinary', 'accuracy'), ('sex-TPR', 'sex-calibration-')]


def run(dataset=get_dataset_names(), graphs=GRAPHS):
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nGenerating graphs for dataset:" + dataset_obj.get_dataset_name())
        for sensitive in dataset_obj.get_sensitive_attributes():
            for tag in TAGS:
                print("    type:" + tag)
                filename = dataset_obj.get_results_filename(sensitive, tag)
                make_all_graphs(filename, graphs)
    print("Generating additional figures in R...")
    subprocess.run(["Rscript", "results/generate-report.R"])


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
    for i, m1 in enumerate(measures):
        for j, m2 in enumerate(measures):
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
        scale = scale_color_manual(values=colors)
        p = (ggplot(f, aes(x=xaxis_measure, y=yaxis_measure, colour='algorithm')) +
             geom_point(size=50) + ggtitle(title) + scale)
        # import ipdb; ipdb.set_trace()
        plt.style.use('ggplot')
        fig, plot = plt.subplots(figsize=(20, 6))
        for i, (name, group) in enumerate(f.groupby('algorithm')):
            plot.plot(group[xaxis_measure], group[yaxis_measure], 'o', label=name, c=colors[i])
        plot.set_xlabel(xaxis_measure)
        plot.set_ylabel(yaxis_measure)
        plot.set_title(title)
        plot.legend()
        fig.savefig(str(save_path / Path(f"pyplot-{xaxis_measure}-{yaxis_measure}.png")))
        print(xaxis_measure, yaxis_measure)
        p.save('results/analysis/%s/%s-%s.png' % (title, xaxis_measure, yaxis_measure),
               width=20,
               height=6)


def generate_rmd_output():
    subprocess.run(["Rscript", "results/generate-report.R"])


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
