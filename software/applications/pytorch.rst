.. _software-applications-pytorch:

PyTorch
-------

`PyTorch <https://pytorch.org/>`__ is an end-to-end machine learning framework.
PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

The main method of distribution for PyTorch is via :ref:`Conda <software-applications-conda>`, with :ref:`Open-CE<software-applications-open-ce>` providing a simple method for installing multiple machine learning frameworks into a single conda environment.

The upstream Conda and pip distributions do not provide ppc64le pytorch packages at this time. 

Installing via Conda
~~~~~~~~~~~~~~~~~~~~

With a working Conda installation (see :ref:`Installing Miniconda<software-applications-conda-installing>`) the following instructions can be used to create a Python 3.9 conda environment named ``torch`` with the latest Open-CE provided PyTorch:

.. note:: 

   Pytorch installations via conda can be relatively large. Consider installing your miniconda (and therfore your conda environments) to the ``/nobackup`` file store.


.. code-block:: bash

   # Create a new conda environment named torch within your conda installation
   conda create -y --name torch python=3.9

   # Activate the conda environment
   conda activate torch

   # Add the OSU Open-CE conda channel to the current environment config
   conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

   # Also use strict channel priority
   conda config --env --set channel_priority strict

   # Install the latest available version of PyTorch
   conda install -y pytorch

In subsequent interactive sessions, and when submitting batch jobs which use PyTorch, you will then need to re-activate the conda environment.

For example, to verify that PyTorch is available and print the version:

.. code-block:: bash

   # Activate the conda environment
   conda activate torch

   # Invoke python
   python3 -c "import torch;print(torch.__version__)"


Installation via the upstream Conda channel is not currently possible, due to the lack of ``ppc64le`` or ``noarch`` distributions.


.. note::
   
   The :ref:`Open-CE<software-applications-open-ce>` distribution of PyTorch does not include IBM technologies such as DDL or LMS, which were previously available via :ref:`WMLCE<software-applications-wmlce>`. 
   WMLCE is no longer supported.


Further Information
~~~~~~~~~~~~~~~~~~~

For more information on the usage of PyTorch, see the `Online Documentation <https://pytorch.org/docs/>`__.
