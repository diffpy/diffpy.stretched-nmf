=============
Release n
0.2.1
=====

**Changed:**

* Rename cli entrypoint to 'snmf' from 'diffpy.stretched-nmf'

**Fixed:**

* Produce an error if test files are missing
* Use matplotlib-base when installing with conda-forge


0.2.0
=====

**Added:**

* Coverage report in each PR
* Implement tests for ``compute_objective_function()``
* Add spelling check pre-commit via Codespell
* Added XRD example using real data
* SNMFOptimizer.converged_ attribute to indicate whether the optimization
successfully reached the convergence tolerance (True) or stopped because the
maximum number of iterations was reached (False).
* L-BFGS-B method in scipy for weight optimization in def get_weights
* Support for Python 3.13
* 'SNMFOptimizer.objective_log' attr: dictionary list to track the optimization
process, recording the step, iteration, objective, and timestamp at each update.
Uses the 'step', 'iteration', 'objective' and 'timestamp' keys.
* 'SNMFOptimizer(verbose : Optional[bool])' option and SNMFOptimizer.verbose
attribute to allow users to toggle diagnostic console output.
* Add docformatter config block to the end of the pyproject.toml file.

**Changed:**

* Refactor ``get_objective_function()`` into a static method and getter
* Move project/package naming and related references to the new ``diffpy.stretched-nmf`` name.
* Update to scikit-package 0.3
* Modified all print messages for improved readability and tied them to the new
verbose flag.
* Refactored convergence checks and step-size calculations to pull objective
values directly from objective_log instead of relying on a separate history
array.

**Fixed:**

* Add getting started section and re-arrange install success check instructions
* Support ``scikit-package`` Level 5 standard (https://scikit-package.github.io/scikit-package/).
* Remove extraneous files
* Register ``pytest.mark.slow`` in pytest config to avoid unknown-marker warnings.
* Add entry point to the application
* Fix CLI bug due to typo
* Reformat README for PyPi compatibility
* Consistently label variables as private or fit-derived
* Absolute tolerance for updated weighted matrix from 0.5 to 1e-05 in test_subroutines.py
* Include GitHub Issues templates for bug report and feature request
* Conform variable names to PEP-8

**Removed:**

* Old tests and source files from prior, pre-release development.
* cvxpy dependency for linear weight optimization in def get_weights
* Support for Python 3.10
* Removed the 'SNMFOptimizer._objective_history' list, which was made redundant
by the comprehensive 'SNMFOptimizer.objective_log' tracking system.

otes
=============

0.1.3
=====

**Fixed:**

* Updated README instructions for pip and conda-forge install
* Updated README instructions to check for successful installation


0.1.2
=====

**Added:**

* Use GitHub Actions to build, release, upload to PyPI
* Added issue template for PyPI/GitHub release

**Changed:**

* Added tag check for release
* citation from arXiv to npj Comput Mater in docs

**Fixed:**

* Python version from 3.9 to 3.12 in CI news item
* tests folder at the root of the repo
* re-cookiecuter repo to groupd's package standard
* Add pip dependencies under pip.txt and conda dependencies under conda.txt


0.1.0
=====

**Added:**

* Initial release of diffpy.snmf

**Changed:**

* Support Python version 3.12
* Remove support for Python version 3.9

**Fixed:**

* Repo structure modified to the new diffpy standard
* Code linting based on .pre-commit-config.yaml
