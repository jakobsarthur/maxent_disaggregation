import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ImportError(
        "plot_samples_hist requires matplotlib. Install it with `pip install matplotlib`."
    ) from exc


def plot_samples_hist(
    samples,
    mean_0=None,
    sd_0=None,
    shares=None,
    sds=None,
    logscale=False,
    plot_agg=True,
    plot_sample_mean=True,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    legend_labels=None,
    save=False,
    filename=None,
):
    """
    Plot histograms of sample distributions, optionally including aggregate and sample means.
    Parameters:
        samples (np.ndarray):
            2D array of shape (n_samples, n_disaggregates) containing the samples to plot.
        mean_0 (float, optional):
            Mean of the aggregate distribution for labeling purposes.
        sd_0 (float, optional):
            Standard deviation of the aggregate distribution for labeling purposes.
        shares (list or np.ndarray, optional):
            List of share values for each disaggregate, used for labeling.
        sds (list or np.ndarray, optional):
            List of standard deviations for each disaggregate, used for labeling.
        logscale (bool, optional):
            If True, use a logarithmic scale for the x-axis. Default is False.
        plot_agg (bool, optional):
            If True, plot the histogram of the aggregate (sum across disaggregates). Default is True.
        plot_sample_mean (bool, optional):
            If True, plot vertical lines for the mean of each sample and the aggregate. Default is True.
        title (str, optional):
            Title for the plot. If None, a default title is used.
        xlabel (str, optional):
            Label for the x-axis. If None, defaults to "Value".
        ylabel (str, optional):
            Label for the y-axis. If None, defaults to "Probability density".
        ylim (tuple, optional):
            Tuple specifying y-axis limits (min, max). If None, limits are set automatically.
        legend_labels (list of str, optional):
            Custom labels for the legend for each disaggregate and the aggregate.
        save (bool, optional):
            If True, save the plot to a file specified by `filename`. Default is False.
        filename (str, optional):
            Path to save the plot if `save` is True.
    Raises:
        ValueError: If `save` is True and `filename` is not provided.
    Notes:
        - Each disaggregate's histogram is plotted with its own color and label.
        - The aggregate histogram (sum across disaggregates) is plotted if `plot_agg` is True.
        - Sample means are indicated with dashed vertical lines if `plot_sample_mean` is True.
        - The function uses matplotlib for plotting and will display the plot unless `save` is True.
    """

    max_height = 0

    for i in range(samples.shape[1]):
        if sds is not None:
            std = sds[i]
        else:
            std = sds
        if shares is not None:
            share = shares[i]
        else:
            share = shares
        if legend_labels is not None:
            label = legend_labels[i]
        else:
            label = f"Share {i+1} input = {share}, SD = {std}"
        x = plt.hist(
            samples[:, i],
            bins=100,
            alpha=0.5,
            label=label,
            density=True,
        )
        max_height = max(max_height, np.percentile(x[0], 100))

        if plot_sample_mean:
            plt.axvline(
                x=samples[:, i].mean(),
                color=x[2][0].get_facecolor(),
                linestyle="--",
                label=f"Share {i+1} sample mean",
            )

    if plot_agg:
        if legend_labels is not None:
            label = legend_labels[-1]
        else:
            label = f"Aggregate input= {mean_0}, SD = {sd_0}"
        x = plt.hist(
            samples.sum(axis=1),
            bins=100,
            alpha=0.5,
            label=label,
            density=True,
        )
        max_height = max(max_height, np.percentile(x[0], 100))
        if plot_sample_mean:
            plt.axvline(
                x=samples.sum(axis=1).mean(),
                color=x[2][0].get_facecolor(),
                linestyle="--",
                label="Aggregate sample mean",
            )

    if logscale:
        plt.xscale("log")

    if not ylim:
        plt.ylim(0, max_height * 1.01)
    else:
        plt.ylim(ylim)

    plt.legend(frameon=True, fontsize=8)
    if xlabel is None:
        xlabel = "Value"
    if ylabel is None:
        ylabel = "Probability density"
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title is None:
        title = "MaxEnt Disaggregation"
    plt.title(title)

    if save:
        if filename is None:
            raise ValueError("Filename must be provided if save is True.")
        plt.savefig(filename)

    plt.show()
