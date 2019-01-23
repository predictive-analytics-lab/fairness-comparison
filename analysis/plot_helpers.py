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


def common_plotting_settings(plot, plot_def, xaxis_title, yaxis_title, legend="inside"):
    """Common settings for plots

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis_title: label for x-axis
        yaxis_title: label for y-axis
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """
    plot.set_xlabel(xaxis_title)
    plot.set_ylabel(yaxis_title)
    if plot_def.title:
        plot.set_title(plot_def.title)
    plot.grid(True)
    if legend == "outside":
        legend = plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
        return legend
    elif legend == "inside":
        plot.legend()
    elif isinstance(legend, tuple) and legend[0] == "outside" and type(legend[1]) == float:
        legend = plot.legend(bbox_to_anchor=(1, legend[1]), loc=2) # , borderaxespad=0.)
        return legend


def scatter(plot, plot_def, xaxis, yaxis, legend="inside", startindex=0, markersize=6):
    """Generate a scatter plot

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """
    shapes = ['o', 'X', 'D', 's', '^', 'v', '<', '>', '*', 'p', 'P']
    colors10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                "#7f7f7f", "#bcbd22", "#17becf"]
    xaxis_measure, yaxis_measure = xaxis[0], yaxis[0]
    filled_counter = startindex
    for i, entry in enumerate(plot_def.entries):
        if entry.do_fill:
            additional_params = dict()
            shp_index = filled_counter
            filled_counter += 1
        else:
            additional_params = dict(mfc='none')
            shp_index = i + startindex - filled_counter
        plot.plot(
            entry.values[xaxis_measure], entry.values[yaxis_measure], shapes[shp_index],
            label=entry.label, **additional_params, c=colors10[shp_index], markersize=markersize
        )
    return common_plotting_settings(plot, plot_def, xaxis[1], yaxis[1], legend)


def errorbox(plot, plot_def, xaxis, yaxis, legend="inside", firstcolor=0, firstshape=0,
             markersize=6):
    """Generate a figure with errorboxes that reflect the std dev of an entry

    Args:
        plot: a pyplot plot object
        plot_def: a `PlotDef` that defines properties of the plot
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        legend: where to put the legend; allowed values: None, "inside", "outside"
    """
    # scale = scale_color_brewer(type='qual', palette=1)
    # d3.schemeCategory20
    # ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    #  "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    #  "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf"]
    pale_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2",
                   "#c7c7c7", "#dbdb8d", "#9edae5"]
    shapes = ['o', 'X', 'D', 's', '^', 'v', '<', '>', '*', 'p', 'P']

    xaxis_measure, yaxis_measure = xaxis[0], yaxis[0]
    filled_counter = firstcolor
    for i, entry in enumerate(plot_def.entries):
        if entry.do_fill:
            color = colors[filled_counter]
            filled_counter += 1
        else:
            color = pale_colors[i + firstcolor - filled_counter]
        i_shp = firstshape + i
        xmean, xstd = np.mean(entry.values[xaxis_measure]), np.std(entry.values[xaxis_measure])
        ymean, ystd = np.mean(entry.values[yaxis_measure]), np.std(entry.values[yaxis_measure])
        plot.bar(xmean, ystd, bottom=ymean - 0.5 * ystd, width=xstd, align='center', color='none',
                 edgecolor=color, linewidth=3, zorder=3 + 2 * i_shp)
        plot.plot(xmean, ymean, shapes[i_shp], c=color, label=entry.label, zorder=4 + 2 * i_shp,
                  markersize=markersize)
    return common_plotting_settings(plot, plot_def, xaxis[1], yaxis[1], legend)


def plot_all(plot_func, plot_def_list, xaxis, yaxis, save=False, legend="inside", figsize=(20, 6),
             dpi=None):
    """Plot all plot definitions in a given list. The plots will be in a single row.

    Args:
        plot_func: which kind of plot to make
        plot_def_list: a list of `PlotDef`
        xaxis: either a string or a tuple of two strings
        yaxis: either a string or a tuple of two strings
        save: True if the figure should be saved to disk
        legend: where to put the legend; allowed values: None, "inside", "outside"
        figsize: size of the whole figure
        dpi: DPI of the figure
    Returns:
        the figure and the array of plots
    """
    # with plt.style.context('seaborn'):  # optional
    if not isinstance(xaxis, tuple):
        xaxis = [xaxis] * 2
    if not isinstance(yaxis, tuple):
        yaxis = [yaxis] * 2
    fig, plots = plt.subplots(ncols=len(plot_def_list), squeeze=False, figsize=figsize)
    legends = []
    for plot, plot_def in zip(plots[0], plot_def_list):
        legend = plot_func(plot, plot_def, xaxis, yaxis, legend)
        if legend is not None:
            legends.append(legend)
    if not save:
        return fig, plots
    save_path = Path("results") / Path("analysis") / Path(plot_def_list[0].title)
    save_path.mkdir(parents=True, exist_ok=True)
    figure_path = str(save_path / Path(f"{xaxis[0]}-{yaxis[0]}.png"))
    if legend == "outside":
        fig.savefig(figure_path, dpi=dpi, bbox_extra_artists=legends, bbox_inches='tight')
    else:
        fig.savefig(figure_path, dpi=dpi)
    # print(xaxis_measure, yaxis_measure)


def parse(filename, filter_transform=None):
    """Parse a file

    You can pass a function as `condition` that decides whether a given algorithm should be
    included. You can pass a function as `mapping` that changes the algorithm names.

    Args:
        filename: a string with the filename
        filter_transform: (optional) a function that takes an algorithm name and returns a
                          replacement name and a boolean that decides if the corresponding marker is
                          filled or returns `None`
    Returns:
        a list of `DataEntry` with algorithm name, Pandas dataframe and fill indicator
    """
    raw_data = pd.read_csv(filename)
    to_plot = []
    for algo_name, values in raw_data.groupby('algorithm'):
        algo_info = (algo_name, True) if filter_transform is None else filter_transform(algo_name)
        if algo_info is not None:
            new_algo_name, do_fill = algo_info
            to_plot.append(DataEntry(label=new_algo_name, values=values, do_fill=do_fill))
    return to_plot


def filter_transform_labels(plot_defs, filter_transform):
    """Filter out and transform entries according to the given function

    Args:
        plot_defs: list of plot definitions
        filter_transform: a function that takes an algorithm name and either returns `None` or a
                          transformed name
    Returns:
        list of `PlotDef` with filtered out entries
    """
    new_plot_defs = []
    for plot_def in plot_defs:
        new_entries = []
        for entry in plot_def.entries:
            algo_info = filter_transform(entry.label)
            if algo_info is not None:
                new_algo_name, do_fill = algo_info
                new_entries.append(entry._replace(label=new_algo_name, do_fill=do_fill))
        new_plot_defs.append(plot_def._replace(entries=new_entries))
    return new_plot_defs


def parse_for_plot(filename, title, filter_transform=None):
    """Parse a file and create a `PlotDef` from it

    Args:
        filename: a string with the filename
        title: title of the plot
        filter_transform: (optional) a function that takes an algorithm name and returns a
                          replacement name and a boolean that decides if the corresponding marker is
                          filled or returns `None`
    Returns:
        a `PlotDef` with a plot title and a list of entries
    """
    return PlotDef(title, parse(filename, filter_transform))


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


def parse_all(filenames_and_titles, filter_transform=None):
    """Parse all given files

    Args:
        filenames_and_titles: a list of tuples with a filename and a title
        filter_transform: (optional) a function that takes an algorithm name and returns a
                          replacement name and a boolean that decides if the corresponding marker is
                          filled or returns `None`
    Returns:
        a list of `PlotDef`, each with a plot title and a list of entries
    """
    return [parse_for_plot(filename, title, filter_transform)
            for filename, title in filenames_and_titles]


def parse_and_plot_all(filenames_and_titles, xaxis, yaxis, filter_transform=None):
    """Parse the given files and then plot the given x- and y-axis"""
    data = parse_all(filenames_and_titles, filter_transform)
    return plot_all(scatter, data, xaxis, yaxis)


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


def choose_entries(plot_defs, new_order):
    """Choose the entries in the plot definitions

    Args:
        plot_defs: list of plot definitions
        new_order: list of indices that define the new order of the entries. Indices may appear
                   multiple times and may appear not at all.
    Returns:
        list of plot definitions with only the chosen entries in the given order
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
