.. _software-python:

Python
======

`Python <https://www.python.org/>`__ is an interpreted, interactive, object-oriented programming language with dynamic typing.

Python 3.6 is available by default on Bede, as ``python3``, however, consider using :ref:`Conda <software-applications-conda>` for your python dependency management.

Conda is a cross-platform package and environment management system, which can provide alternate python versions than distributed centrally, and is more-suitable for managing packages which include non-python dependencies. 

Python 2 is also available, but is no longer an officially supported version of python. 
If you are still using python 2, upgrade to python 3 as soon as possible.

.. note::

    The ``python`` executable refers to ``python2`` on RHEL 7, but ``python3`` on RHEL 8 images. Consider using the more specific ``python3`` command.

If you wish to use non-conda python, you should use `virtual environments <https://docs.python.org/3/library/venv.html>`__ to isolate your python environment(s) from the system-wide environment.
This will allow you to install your own python dependencies via pip.


For instance, to create and install `sphinx` (the python package used to create this documentation) into a python environment in your home directory:

.. code-block:: bash

   # Create a directory for your venvs if it does not exist
   mkdir -p ~/.venvs
   # Create a python3 venv named sphinx, located at ~/.venvs/sphinx
   python3 -m venv ~/.venvs/sphinx
   # Activate the virtual environment. You will need to do this any time you with to use the environment
   source ~/.venvs/sphinx/bin/activate
   # Verify the location of your python3
   which python3
   # Use pip to install sphinx into the environment
   python3 -m pip install sphinx

.. note::
  
   Python virtual environments can become large if large python packages such as TensorFlow are installed. 
   Consider placing your python virtual environments in your project directories to avoid filling your home directory.


Python virtual environments can be deactivated using the ``deactivate`` command

.. code-block:: bash

   deactivate

They can be deleted by recursively deleting the directory.

I.e. to delete a python virtual environment located at ``~/.venvs/sphinx``

.. code-block:: bash

    rm -r ~/.venvs/sphinx/


For further information on please see the `Python Online Documentation <https://docs.python.org/3/index.html>`__.
