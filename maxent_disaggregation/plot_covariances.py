try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ImportError(
        "plot_covariances requires matplotlib. Install it with `pip install matplotlib`."
    ) from exc
try:
    import corner
except ModuleNotFoundError as exc:
    raise ImportError(
        "plot_covariances requires corner. Install it with `pip install corner`."
    ) from exc


def plot_covariances(
    samples,
    title=None,
    labels=None,
    save=False,
    filename=None,
):
    """
    Plot the covariance matrix of the samples using corner plots.
    Parameters
    ----------
        samples : np.ndarray
            2D array of shape (n_samples, n_disaggregates) containing the samples to plot.
        title : str, optional
            Title for the plot. If None, a default title is used.
        labels : list of str, optional
            Custom labels for each disaggregate. If None, defaults to "Share 1", "Share 2", etc.
        save : bool, optional
            If True, save the plot to a file specified by `filename`. Default is False.
        filename : str, optional
            Path to save the plot if `save` is True.
    Raises
    ------
        ValueError: If `save` is True and `filename` is not provided.

    Notes
    -----
    - The function uses the `corner` library to create a corner plot of the samples.
    """

    if not labels:
        labels = [f"Share {i+1}" for i in range(samples.shape[1])]

    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        smooth=True,
        smooth1d=True,
        fill_contours=False,
        levels=(0.68, 0.95),
        bins=50,
        plot_datapoints=True,
        color="C0",
    )

    if title is None:
        title = "MaxEnt Disaggregation Covariances"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save:
        if filename is None:
            raise ValueError("Filename must be provided if save is True.")
        plt.savefig(filename)

    plt.show()
