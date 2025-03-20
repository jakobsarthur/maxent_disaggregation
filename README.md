# maxent_disaggregation

[![PyPI](https://img.shields.io/pypi/v/maxent_disaggregation.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/maxent_disaggregation.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/maxent_disaggregation)][pypi status]
[![License](https://img.shields.io/pypi/l/maxent_disaggregation)][license]

[![Read the documentation at https://maxent_disaggregation.readthedocs.io/](https://img.shields.io/readthedocs/maxent_disaggregation/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/jakobsarthur/maxent_disaggregation/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/jakobsarthur/maxent_disaggregation/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/maxent_disaggregation/
[read the docs]: https://maxent_disaggregation.readthedocs.io/
[tests]: https://github.com/jakobsarthur/maxent_disaggregation/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/jakobsarthur/maxent_disaggregation
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Installation

You can install _maxent_disaggregation_ via [pip] from [PyPI]:

```console
$ pip install maxent_disaggregation
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide][Contributor Guide].

## License

Distributed under the terms of the [MIT license][License],
_maxent_disaggregation_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue][Issue Tracker] along with a detailed description.


<!-- github-only -->

[command-line reference]: https://maxent_disaggregation.readthedocs.io/en/latest/usage.html
[License]: https://github.com/jakobsarthur/maxent_disaggregation/blob/main/LICENSE
[Contributor Guide]: https://github.com/jakobsarthur/maxent_disaggregation/blob/main/CONTRIBUTING.md
[Issue Tracker]: https://github.com/jakobsarthur/maxent_disaggregation/issues


## Building the Documentation

You can build the documentation locally by installing the documentation Conda environment:

```bash
conda env create -f docs/environment.yml
```

activating the environment

```bash
conda activate sphinx_maxent_disaggregation
```

and [running the build command](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#sphinx-build):

```bash
sphinx-build docs _build/html --builder=html --jobs=auto --write-all; open _build/html/index.html
```