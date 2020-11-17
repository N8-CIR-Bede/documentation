.. _bede_pytorch:

PyTorch
=======

.. sidebar:: PyTorch

   :URL: https://pytorch.org

PyTorch is an open source machine learning library for Python, based on `Torch <http://torch.ch/>`_.
It is used for applications such as natural language processing.

About PyTorch on Bede
---------------------

.. note::
   GPU must be requested in order to enable GPU acceleration by adding the flag e.g. ``--gpus=1`` to the scheduler command or job script.
   See :ref:`bede_scheduler` for more information.

Bede has a locally installed IBM `Watson Machine Learning Community Edition (WML CE) <https://developer.ibm.com/linuxonpower/deep-learning-powerai/releases/>`_ Anaconda channel that provides versions of Tensorflow, PyTorch and their dependencies especially built for the POWER architecture.

Installation in Home Directory
------------------------------

First request an interactive session, e.g. see :ref:`bede_scheduler`.

Then PyTorch can be installed by the following ::

   # Load the conda module
   module load Anaconda3/2020.02

   # Add the local WML CE channel to the conda search path
    conda config --prepend channels file:///opt/software/apps/ibm_wmlce/wmlce-1.7.0-mirror/

   # Create an conda virtual environment called e.g. named 'pytorch'
   conda create -n pytorch python=3.6

   # Activate the 'pytorch' environment
   source activate pytorch

   # Install PyTorch
   conda install torch


**Every Session Afterwards and in Your Job Scripts**

Every time you use a new session or within your job scripts,
the modules must be loaded and conda must be activated again.
Use the following command to activate the Conda environment with PyTorch installed: ::

   # Load the conda module
   module load Anaconda3/2020.02
   
   # Activate the 'pytorch' environment
   source activate pytorch

Testing your PyTorch installation
---------------------------------

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code
e.g. an example from the official PyTorch `getting started <https://pytorch.org/get-started/locally/>`_ guide
(replicated below).

Here we construct a randomly-initialized tensor: ::

  import torch
  x = torch.rand(5, 3)
  print(x)

The output should be something similar to: ::

   tensor([[0.3380, 0.3845, 0.3217],
           [0.8337, 0.9050, 0.2650],
           [0.2979, 0.7141, 0.9069],
           [0.1449, 0.1132, 0.1375],
           [0.4675, 0.3947, 0.1426]])

Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch,
run the following commands to return whether or not the CUDA driver is enabled: ::

   import torch
   torch.cuda.is_available()
