=============
Release notes
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
