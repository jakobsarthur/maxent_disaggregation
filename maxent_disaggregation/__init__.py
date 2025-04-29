"""maxent_disaggregation."""

from .shares import sample_shares
from .maxent_disaggregation import maxent_disagg
from .maxent_disaggregation import sample_aggregate


__all__ = (
    "__version__",
    "maxent_disagg",
    "sample_shares",
    "sample_aggregate",
    # Add functions and variables you want exposed in `maxent_disaggregation.` namespace here
)

__version__ = "0.0.2.1"
