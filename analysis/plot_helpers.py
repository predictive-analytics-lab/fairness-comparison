"""
Functions that are helpful for plotting results
"""

from pathlib import Path
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# data that will appear as one entry in the legend of a plot
DataEntry = namedtuple('DataEntry', ['label', 'values', 'do_fill'])

# definition of a plot with a list of entries and a title
PlotDef = namedtuple('PlotDef', ['title', 'entries'])


def general_plotting(plot, plot_fun, plot_def, xaxis, yaxis, legend_outside):
    """General function that can be used to plot data in many different ways

    Args:
        plot: a pyplot plot object
        plot_fun: function that handles the plotting
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend_outside: True if the legend should be outside of the plots
    """
    if isinstance(xaxis, tuple):
        xaxis_measure, xaxis_title = xaxis
    else:
        xaxis_measure, xaxis_title = [xaxis] * 2
    if isinstance(yaxis, tuple):
        yaxis_measure, yaxis_title = yaxis
    else:
        yaxis_measure, yaxis_title = [yaxis] * 2

    plot_fun(plot, plot_def, xaxis_measure, yaxis_measure)
    plot.set_xlabel(xaxis_title)
    plot.set_ylabel(yaxis_title)
    plot.set_title(plot_def.title)
    plot.grid()
    if legend_outside:
        legend = plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
        return (xaxis_measure, yaxis_measure), legend
    plot.legend()
    return (xaxis_measure, yaxis_measure), None


def generate_graph(plot, plot_def, xaxis, yaxis, legend_outside=False):
    """Generate a single plot

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend_outside: True if the legend should be outside of the plots
    """
    shapes = ['o', 'D', 's', '*', '^', 'v', '<', '>', 'p', 'X', 'P']

    def _core_plot(plot, plot_def, xaxis_measure, yaxis_measure):
        filled_counter = 0
        for i, entry in enumerate(plot_def.entries):
            if entry.do_fill:
                additional_params = dict()
                shp_index = filled_counter
                filled_counter += 1
            else:
                additional_params = dict(mfc='none')
                shp_index = i - filled_counter
            plot.plot(
                entry.values[xaxis_measure], entry.values[yaxis_measure], shapes[shp_index],
                label=entry.label, **additional_params  # , c=colors[i]
            )
    return general_plotting(plot, _core_plot, plot_def, xaxis, yaxis, legend_outside)


def errorbox(plot, plot_def, xaxis, yaxis, legend_outside=False):
    """Generate a figure with errorboxes that reflect the std dev of an entry

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend_outside: True if the legend should be outside of the plots
    """
    # scale = scale_color_brewer(type='qual', palette=1)
    # d3.schemeCategory20
    # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

    # colors20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728",
    #             "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2",
    #             "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
    colors10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22", "#17becf"]

    def _core_plot(plot, plot_def, xaxis_measure, yaxis_measure):
        for i, entry in enumerate(plot_def.entries):
            xmean, xstd = np.mean(entry.values[xaxis_measure]), np.std(entry.values[xaxis_measure])
            ymean, ystd = np.mean(entry.values[yaxis_measure]), np.std(entry.values[yaxis_measure])
            plot.bar(xmean, ystd, bottom=ymean - 0.5 * ystd, width=xstd, align='center',
                     color='none', edgecolor=colors10[i], label=entry.label, linewidth=3,
                     zorder=1000.)
            plot.plot(xmean, ymean, 'ko')
    return general_plotting(plot, _core_plot, plot_def, xaxis, yaxis, legend_outside)


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


def parse_for_plot(filename, title, condition=None, mapping=None):
    """Parse a file and create a `PlotDef` from it

    Args:
        filename: a string with the filename
        title: title of the plot
        condition: (optional) a function that takes an algorithm name
                   and returns True or False (i.e. a predicate)
        mapping: (optional) a function that takes an algorithm name
                 and returns a replacement name and a boolean that decides
                 if the corresponding marker is filled or not
    Returns:
        a `PlotDef` with a plot title and a list of entries
    """
    return PlotDef(title, parse(filename, condition, mapping))


def transform(entries, key, transformation):
    """Transform a column in a DataEntry

    Args:
        entries: list of `DataEntry`
        key: a string that identifies a column in the pandas dataframes
        transformation: a function that takes a value and returns some of it
    Returns:
        a list of `DataEntry` where the one given column has been transformed in all entries
    """
    new_entries = []
    for entry in entries:
        values = entry.values
        values[key] = values[key].map(transformation)
        new_entries.append(entry._replace(values=values))
    return new_entries


def transform_plot_def(plot_def, key, transformation):
    """Transform the entries in a `PlotDef`"""
    return plot_def._replace(entries=transform(plot_def.entries, key, transformation))


def transform_all(plot_def_list, key, transformation):
    """Apply transformation to all entries in all given plot definitions"""
    new_plot_defs = []
    for plot_def in plot_def_list:
        new_plot_defs.append(transform_plot_def(plot_def, key, transformation))
    return new_plot_defs


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
        a list of `PlotDef`, each with a plot title and a list of entries
    """
    return [parse_for_plot(filename, title, condition, mapping)
            for filename, title in filenames_and_titles]


def parse_and_plot_all(filenames_and_titles, xaxis, yaxis, condition=None, mapping=None):
    """Parse the given files and then plot the given x- and y-axis"""
    data = parse_all(filenames_and_titles, condition, mapping)
    return plot_all(generate_graph, data, xaxis, yaxis)


def plot_all(plot_func, plot_def_list, xaxis, yaxis, save=False, legend_outside=False,
             figsize=(20, 6), dpi=None):
    """Plot all plot definitions in a given list. The plots will be in a single row.

    Args:
        plot_func: which kind of plot to make
        plot_def_list: a list of `PlotDef`
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        save: True if the figure should be saved to disk
        legend_outside: True if the legend should be outside of the plots
        figsize: size of the whole figure
        dpi: DPI of the figure
    Returns:
        the figure and the array of plots
    """
    # with plt.style.context('seaborn'):  # optional
    fig, plots = plt.subplots(ncols=len(plot_def_list), squeeze=False, figsize=figsize)
    legends = []
    for plot, plot_def in zip(plots[0], plot_def_list):
        axes, legend = plot_func(plot, plot_def, xaxis, yaxis, legend_outside)
        if legend is not None:
            legends.append(legend)
    if not save:
        return fig, plots
    save_path = Path("results") / Path("analysis") / Path(plot_def_list[0].title)
    save_path.mkdir(parents=True, exist_ok=True)
    figure_path = str(save_path / Path(f"{axes[0]}-{axes[1]}.png"))
    if legend_outside:
        fig.savefig(figure_path, dpi=dpi, bbox_extra_artists=legends, bbox_inches='tight')
    else:
        fig.savefig(figure_path, dpi=dpi)
    # print(xaxis_measure, yaxis_measure)


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


def merge_same_labels(plot_defs):
    """Merge entries that have the same label"""
    new_plot_defs = []
    for plot_def in plot_defs:
        new_entries = []
        entry_index = {}  # for finding the right index in `new_entries`
        for entry in plot_def.entries:
            if entry.label not in entry_index:
                # new entry
                entry_index[entry.label] = len(new_entries)
                # we store a tuple of the entry and a list of values
                new_entries.append((entry, [entry.values]))
            else:
                # find the index for this label
                ind = entry_index[entry.label]
                # append the values to the list of values (second entry of the tuple)
                new_entries[ind][1].append(entry.values)

        # convert the list of tuples to a list of entries, in place
        for j, (entry, list_of_values) in enumerate(new_entries):
            # `pd.concat` merges the list of dataframes
            new_entries[j] = entry._replace(values=pd.concat(list_of_values))
        new_plot_defs.append(plot_def._replace(entries=new_entries))
    return new_plot_defs


def reorder_entries(plot_defs, new_order):
    """Reorder the entries in the plot definitions

    Args:
        plot_defs: list of plot definitions
        new_order: list of indices that define the new order of the entries. Indices may appear
                   multiple times and may appear not at all.
    Returns:
        list of plot definitions with reordered entries
    """
    new_plot_defs = []
    for plot_def in plot_defs:
        new_plot_defs.append(plot_def._replace(entries=[plot_def.entries[i] for i in new_order]))
    return new_plot_defs


def merge_entries(base_plot_def, *additional_plot_defs):
    """Merge the entries of the given plot definitions

    Args:
        base_plot_def: the merged plot definition will have the title of this one
        additional_plot_defs: any number of additional plot definitions
    Returns:
        plot definitions that has all the entries of the given plot definitions
    """
    # copy in order not to change the given base_plot_def
    all_entries = base_plot_def.entries.copy()
    for plot_def in additional_plot_defs:
        all_entries += plot_def.entries
    return base_plot_def._replace(entries=all_entries)


def merge_plot_defs(base_plot_def_list, *plot_def_lists):
    """Merge all plot definitions in the given list

    This function expects several lists that all have the same length and all contain plot
    definitions. The plot definitions are then merged to produce a single list that has the same
    length as the input lists.

    Args:
        base_plot_def_list: list of plot definitions that are used as the base
        plot_def_lists: any number of additional plot definition lists
    Returns:
        a single list of plot definitions where each plot definition has all the entries of the
        corresponding plot definitions in the other lists
    """
    merged_plot_def_list = []
    # iterate over all given lists simultaneously
    for plot_defs in zip(base_plot_def_list, *plot_def_lists):
        base_plot_def = plot_defs[0]
        # make sure the plot definitions that we are trying to merge are actually compatible
        # we do this by checking whether the titles are all the same
        for plot_def in plot_defs:
            if base_plot_def.title != plot_def.title:
                raise ValueError("Titles have to be the same everywhere")
        merged_plot_def_list.append(merge_entries(base_plot_def, *plot_defs[1:]))
    return merged_plot_def_list
