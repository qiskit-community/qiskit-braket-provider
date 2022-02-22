File descriptions for template
==============================

- [.github](../.github) - folder for configuring anything related to github. 
  For example how issues and pull requests are looking like or CI processes setup (automatic tests, style checks).
  Currently, we have
    - [templates for issues](../.github/ISSUE_TEMPLATE)
    - [pull requests template](../.github/PULL_REQUEST_TEMPLATE.md)
    - [CI processes](../.github/workflows):
        - [tests](../.github/workflows/tests.yml)
        - [code style checks](../.github/workflows/lint.yml)
        - [test coverage checks](../.github/workflows/coverage.yml)
- [.gitignore](../.gitignore) - GitHub specific file that tells which files to ignore
  when tracking/pushing/commiting code (those files will not be tracked by github)
- [.pylintrc](../.pylintrc) - standard style checks configuration file. Contend of file is 
  self-explanatory. During automatic style checks CI processes are referring to this file
  to get guidelines.
- [.travis.yml](../.travis.yml) - for internal repositories we use Travis - CI framewok. 
  This is similar framework to GitHub actions which are described in [CI processes](../.github/workflows).
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - one of the standard requirements for GitHub repositories. 
  Name speak for itself.
- [CONTRIBUTING.md](../CONTRIBUTING.md) - one of the standard requirements for GitHub repositories. 
  Contributing guidelines for developers.
- [LICENSE.txt](../LICENSE.txt) - one of the standard requirements for GitHub repositories. 
  There are different types of [licensing software](https://en.wikipedia.org/wiki/Software_license).
  [Most popular open-source licenses](https://opensource.org/licenses).
- [README.md](../README.md) - main readme for repository.
- [docs](../docs) - documentation for repository.
- [requirements.txt](../requirements.txt) - list of required 3rd party packages to run your project.
- [requirements-dev.txt](../requirements-dev.txt) - list of required 3rd party packages that are 
  NOT required to run your project, but might benefit future developers. It can include specific test
  libraries, style checks packages etc.
- [setup.cfg](../setup.cfg) - configuration metadata for project.
- [setup.py](../setup.py) - file that tells package managers how to use your project. 
  This is the main configuration file for all Python projects.
- [tests](../tests) - folder where all project tests are located. 
  It is a good practice to cover your project with tests to ensure correctness of implementation.
- [tox.ini](../tox.ini) - configuration file for [tox](https://tox.readthedocs.io/en/latest/) framework that 
  aims to automate and standardize testing in Python. 
  Eases the packaging, testing and release process of Python software.


