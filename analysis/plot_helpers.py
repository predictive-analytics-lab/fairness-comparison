"""
Functions that are helpful for plotting results
"""

from pathlib import Path
from collections import namedtuple

from matplotlib import pyplot as plt
import pandas as pd

DataEntry = namedtuple('DataEntry', ['label', 'values', 'do_fill'])
PlotDef = namedtuple('PlotDef', ['title', 'entries'])


def generate_graph(data_list, xaxis, yaxis, save=False, legend_outside=False, figsize=(20, 6),
                   dpi=None):
    """Generate a figure with multiple plots

    Args:
        data_list: a list of `PlotDef`
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        save: True if the figure should be saved to disk
        legend_outside: True if the legend should be outside of the plots
        figsize: size of the whole figure
        dpi: DPI of the figure
    """
    if isinstance(xaxis, tuple):
        xaxis_measure, xaxis_title = xaxis
    else:
        xaxis_measure, xaxis_title = [xaxis] * 2
    if isinstance(yaxis, tuple):
        yaxis_measure, yaxis_title = yaxis
    else:
        yaxis_measure, yaxis_title = [yaxis] * 2

#     scale = scale_color_brewer(type='qual', palette=1)
#     d3.schemeCategory20
#     ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
#      "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
#      "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
#     colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
#               "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
#               "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
#               "#17becf", "#9edae5"]

    shapes = ['o', 'D', 's', '*', '^', 'v', '<', '>', 'p', 'X', 'P']

    # with plt.style.context('seaborn'):  # optional
    fig, plots = plt.subplots(ncols=len(data_list), squeeze=False, figsize=figsize)
    legends = []
    for plot, plot_def in zip(plots[0], data_list):
        filled_counter = 0
        for i, entry in enumerate(plot_def.entries):
            if entry.do_fill:
                additional_params = dict()
                shp_index = filled_counter
                filled_counter += 1
            else:
                additional_params = dict(mfc='none')
                shp_index = i - filled_counter
            plot.plot(entry.values[xaxis_measure], entry.values[yaxis_measure], shapes[shp_index],
                      label=entry.label, **additional_params  # c=colors[i],
            )
        plot.set_xlabel(xaxis_title)
        plot.set_ylabel(yaxis_title)
        plot.set_title(plot_def.title)
        plot.grid()
        if legend_outside:
            legends.append(plot.legend(loc='upper left', bbox_to_anchor=(1, 1)))
        else:
            plot.legend()
    if save:
        save_path = Path("results") / Path("analysis") / Path(data_list[0].title)
        save_path.mkdir(parents=True, exist_ok=True)
        figure_path = str(save_path / Path(f"{xaxis_measure}-{yaxis_measure}.png"))
        if legend_outside:
            fig.savefig(figure_path, dpi=dpi, bbox_extra_artists=legends, bbox_inches='tight')
        else:
            fig.savefig(figure_path, dpi=dpi)
        # print(xaxis_measure, yaxis_measure)


def parse(filename, condition=None, mapping=None):
    """Parse a file

    You can pass a function as `condition` that decides whether a given
    algorithm should be included.
    You can pass a function as `mapping` that changes the algorithm names

    Args:
        filename: a string with the filename
        condition: (optional) a function that takes an algorithm name
                   and returns True or False (i.e. a predicate)
        mapping: (optional) a function that takes an algorithm name
                 and returns a replacement name and a boolean that decides
                 if the corresponding marker is filled or not
    Returns:
        a list of `DataEntry` with algorithm name, Pandas dataframe and fill indicator
    """
    no_cond = (condition is None)
    no_map = (mapping is None)
    raw_data = pd.read_csv(filename)
    to_plot = []
    for algo_name, values in raw_data.groupby('algorithm'):
        if no_cond or condition(algo_name):
            new_algo_name, do_fill = (algo_name, True) if no_map else mapping(algo_name)
            to_plot.append(DataEntry(label=new_algo_name, values=values, do_fill=do_fill))
    return to_plot


def transform(entries, key, transformation):
    """Transform a column in a DataEntry"""
    new_entries = []
    for entry in entries:
        values = entry.values
        values[key] = values[key].map(transformation)
        new_entries.append(entry._replace(values=values))
    return new_entries


def parse_all(filenames_and_titles, condition=None, mapping=None):
    """Parse all given files

    Args:
        filenames_and_titles: a list of tuples with a filename and a title
        condition: (optional) a function that takes an algorithm name
                   and returns True or False (i.e. a predicate)
        mapping: (optional) a function that takes an algorithm name
                 and returns a replacement name and a boolean that decides
                 if the corresponding marker is filled or not
    Returns:
        a list of `PlotDef` with a plot title and a list of entries
    """
    return [PlotDef(title, parse(filename, condition, mapping))
            for filename, title in filenames_and_titles]


def plot_all(filenames_and_titles, xaxis, yaxis, condition=None, mapping=None):
    """Plot the given x- and y-axis of all given files"""
    data = parse_all(filenames_and_titles, condition, mapping)
    generate_graph(data, xaxis, yaxis)


def start_filter(startswith):
    """Return a filter that lets strings through that start with `startswith`"""
    def _filt(label):
        return label.startswith(startswith)
    return _filt


def mark_as_unfilled(startswith):
    """Mark all entries as unfilled where the label begins with `startswith`"""
    def _mapping(label):
        return label, not label.startswith(startswith)
    return _mapping
