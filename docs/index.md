
# maxent_disaggregation
---------------------------------------------

💡 maxent_disaggregation is a [![Python Package](https://img.shields.io/badge/Python-Package-blue?style=for-the-badge&logo=python)](https://github.com/jakobsarthur/maxent_disaggregation) designed to help with uncertainty propagation in the case of disaggregating data. More in general, the package can be used for correlated sampling of shares (quanitites that sum up to a fixed but possible uncertain total), even if only limited or
partial information of the data is availaible. The package closely follows the ``Maximum Entropy`` principle, 
to provide unbiased sampling, using all available information. Sounds complicated? The package does the hard work for you! Simply put in the information that you have and sample away!

Check out these docs to:

- Have a [Quick Start](./content/usage.md)
- See [How it works...](./content/theory.md)
- See [Example notebooks](./content/examples/index.md)


➡ Click here for the [![R Package](https://img.shields.io/badge/R-Package-orange?style=for-the-badge&logo=r)](https://github.com/simschul/MaxentDisaggregation) of the package!



## ✨ Features
This package enables you to account for:
- **Disaggregate data** into multiple data via estimates of the relative shares
- **Propagate the uncertainty** from the raw data to the disaggregate samples (even in the case of missing data)
- **Preserve statistical correlation** due to disaggregating data
- **Plot the sampled distributions** with a one function call
- **Plot the covariance** of the samples



## 💬 Support
If you have any questions or need help, do not hesitate to contact us:
- Arthur Jakobs ([artos.jakobs@psi.ch](mailto:artos.jakobs@psi.ch))
- Simon Schulte ([simonsc@plan.aau.dk](mailto:simonsc@plan.aau.dk))





```{toctree}
---
hidden:
maxdepth: 1
---

Installation <content/installation>
Usage <content/usage>
Theory <content/theory>
Examples <content/examples/index>
API <content/api/index>
Code of Conduct <content/codeofconduct>
Contributing <content/contributing>
License <content/license>
Changelog <content/changelog>
```
