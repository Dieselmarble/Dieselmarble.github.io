---
title: Create a virtual python environment
date: 2018-10-31 20:39:33
tags:
categories: Blog
visible: 
---

## Create a virtual python environment 


Python applications will often use packages and modules that don’t come as part of the standard library. Applications will sometimes need a specific version of a library, because the application may require that a particular bug has been fixed or the application may be written using an obsolete version of the library’s interface.

This means it may not be possible for one Python installation to meet the requirements of every application. If application A needs version 1.0 of a particular module but application B needs version 2.0, then the requirements are in conflict and installing either version 1.0 or 2.0 will leave one application unable to run.

The solution for this problem is to create a virtual environment, a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages.

> https://docs.python.org/3/tutorial/venv.html

```
virtualenv --version                  # If a version number is not output, see https://virtualenv.pypa.io/en/stable/installation/.
which python                          # If the 'python' command is aliased to something like '/usr/bin/python27', prepare to unalias it.
unalias python                        # If the 'python' command is aliased to something like '/usr/bin/python27', unalias it.
python --version                      # Output the current Python version, for example 'Python 2.7.12'.
python3 --version                     # Output the current version of Python 3, for example 'Python 3.6.2'.
which python36                        # Output the path to the python36 binary, for example '/usr/bin/python36'.
cd ~/environment/                     # Prepare to create a virtual environment in this path.
virtualenv -p /usr/bin/python36 vpy36 # Create a virtual environment for Python 3.6 in this path.
source vpy36/bin/activate             # Switch to use Python 3.6 instead of Python 2.7.12 when you run the 'python --version' command.
python --version                      # Output the current Python version, for example 'Python 3.6.2'.
deactivate                            # If and when you are done using Python 3.6, prepare to make Python 2.7.12 active again.
alias python=/usr/bin/python27        # Switch back to outputting '/usr/bin/python27' when you run the 'which python' command.
```