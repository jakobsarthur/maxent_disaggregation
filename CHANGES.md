# `maxent_disaggregation` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [1.3.2]
* always import plotting dependencies

## [1.3.1]
* Added optional `seed` parameter to `maxent_disagg`, `sample_shares`, and `sample_aggregate` (and their internal helpers) for reproducible random sampling.

## [1.3.0]
* moved plotting functionality into dedicated modules.
* made plotting dependencies check at plotting-function import, this removes the hard dependencies on matplotlib and corner.

## [1.2.0]
* removed explicit print statements from warning catch and added suppress_warnings flag to suppress warnings*


## [1.1.0]
* integrated an unbiased truncated normal, through least squares optimization of the Gausian paramteres of the truncatd normal distribution.
* included simple tests



## [1.0.6]
* update dependencies, all plotting dependencies only in the notebook environment

## [1.0.1]
* Added some check on inputs.

## [1.0.0] - 2025-06-10
* Paper resubmission release. Updated Hybrid Dirichlet --> hybrid Dirichlet (a compbination of Beta distributions and maximum entropy dirichlet distribution). Bias checks are performed. Includes plotting functions of distribution histograms and covariances. 


## [0.0.2] - 2025-04-10
* Incorporated nested sampling in mixed information case. Also testing for the mean of the sampled shares vs the provided shares and warns if above a threshold value. Same for the standatrd deviations on the mean.

### Added

### Changed

### Removed
