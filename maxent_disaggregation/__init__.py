"""maxent_disaggregation."""

from typing import TYPE_CHECKING

from .shares import sample_shares
from .aggregate import sample_aggregate
from .maxent_disaggregation import maxent_disagg
from .maxent_direchlet import find_gamma_maxent

if TYPE_CHECKING:
    from .plot_covariances import plot_covariances
    from .plot_samples_hist import plot_samples_hist


__all__ = (
    "__version__",
    "maxent_disagg",
    "sample_shares",
    "sample_aggregate",
    "plot_samples_hist",
    "plot_covariances",
    "find_gamma_maxent",
    # Add functions and variables you want exposed in `maxent_disaggregation.` namespace here
)


def __getattr__(name):
    if name == "plot_samples_hist":
        try:
            from .plot_samples_hist import plot_samples_hist as func
        except ImportError as exc:
            raise ImportError(
                "plot_samples_hist requires plotting dependencies. "
                "Install with `pip install matplotlib`."
            ) from exc

        # Importing the submodule sets `plot_samples_hist` on this package's
        # namespace to the *module*, shadowing this __getattr__ on future
        # lookups. Overwrite it with the actual function so both attribute
        # access and `from maxent_disaggregation import plot_samples_hist`
        # resolve correctly.
        globals()[name] = func
        return func
    if name == "plot_covariances":
        try:
            from .plot_covariances import plot_covariances as func
        except ImportError as exc:
            raise ImportError(
                "plot_covariances requires plotting dependencies. "
                "Install with `pip install matplotlib corner`."
            ) from exc

        globals()[name] = func
        return func
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__version__ = "1.3.3"
