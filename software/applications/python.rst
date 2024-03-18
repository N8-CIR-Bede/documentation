.. _software-python:

Python
======

`Python <https://www.python.org/>`__ is an interpreted, interactive, object-oriented programming language with dynamic typing.

.. tabs::

   .. tab:: ppc64le 

         Python ``3.6`` is available by default, as ``python3``, however, consider using :ref:`Conda <software-applications-conda>` for your python dependency management.

         Conda is a cross-platform package and environment management system, which can provide alternate python versions than distributed centrally, and is more-suitable for managing packages which include non-python dependencies. 

         On the ``ppc64le`` nodes/partitions  Python 2 is also available, but is no longer an officially supported version of python. 
         If you are still using python 2, upgrade to python 3 as soon as possible.

   .. tab:: aarch64

      Python ``3.9`` is available by default on ``aarch64`` nodes, as ``python3`` and ``python``.
      Alternate versions of Python can be installed via :ref:`Conda <software-applications-conda>` 

      Conda is a cross-platform package and environment management system, which can provide alternate python versions than distributed centrally, and is more-suitable for managing packages which include non-python dependencies. 

      Python 2 is not available on the ``aarch64`` nodes/partitions in Bede.
 
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

.. warning::
  
   Python virtual environments can become large if large python packages such as TensorFlow are installed. 
   Consider placing your python virtual environments in your project directories to avoid filling your home directory.


Python virtual environments can be deactivated using the ``deactivate`` command

.. code-block:: bash

   deactivate

They can be deleted by recursively deleting the directory.

I.e. to delete a python virtual environment located at ``~/.venvs/sphinx``

.. code-block:: bash

    rm -r ~/.venvs/sphinx/


Python packages may install architecture dependent binaries, so you should use a separate virtual environments for ``ppc64le`` and ``aarch64`` nodes/partitions.

For further information on please see the `Python Online Documentation <https://docs.python.org/3/index.html>`__.
