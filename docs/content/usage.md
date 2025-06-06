# Quickstart

After installation use the following example as a way to quickly guide you through the usage of the `maxent_disaggregation` package. For more detailed examples please see [Examples](./examples/index.md)


```python
from maxent_disaggregation import maxent_disagg
import numpy as np

# best guess or mean of the total quantity Y_0 (if available)
mean_aggregate = 10
# best guess of the standard deviation of the total quantity Y_0 (if available)
sd_aggregate = 1
# min/max value of the total quantity Y_o (if applicable/available) (optional)
min_aggregate = 0
max_aggregate = np.inf
# best guess values and uncertainties from proxy data for the shares (x_i) if available (of not available put in np.nan)
shares_disaggregates = [0.4, 0.25, 0.2, 0.15]
sds_shares = [0.1, np.nan, 0.04, 0.001]



# Now draw 10000 samples
samples, _ = maxent_disagg(n=10000, 
                        mean_0=mean_aggregate,
                        sd_0=sd_aggregate,
                        min_0=min_aggregate,
                        max_0=max_aggregate, 
                        shares=shares_disaggregates, 
                        sds=sds_shares, 
                        )

# Now plot the sampled distributions
from maxent_disaggregation import plot_samples_hist
# the input values are provided for the legend
plot_samples_hist(samples, 
                  mean_0=mean_aggregate,
                  sd_0=sd_aggregate, 
                  shares=shares_disaggregates, 
                  sds=sds_shares)
```
```{figure} data/Quickstart_example.svg
:align: center
:alt: Histograms of the samples for both the disaggregate and aggregate quantities.
Figure 1: Histograms of the samples for both the disaggregate and aggregate values. The dashed vertical lines indicate the means of the sampled distributions. The input values are given in the legend. 
```


We can also easily plot the covariances between the different disaggrate quantities:

```python
# Plot the covariances between the disaggregates
from maxent_disaggregation import plot_covariances
plot_covariances(samples)
```

```{figure} data/Quickstart_example_covariances.svg
:align: center
:alt: Covariances of the samples for the disaggregate quantities.
Figure 2: Covariances of the samples for the disaggregate quantities. The dashed vertical lines indicate the means and $\pm 1\sigma$ of the sampled distributions. The input values are given in the legend. 
```
