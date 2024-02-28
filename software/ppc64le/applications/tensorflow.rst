.. _software-applications-tensorflow:

TensorFlow
----------

`TensorFlow <https://www.tensorflow.org/>`__ is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

TensorFlow can be installed through a number of python package managers such as :ref:`Conda<software-applications-conda>` or ``pip``.

For use on Bede, the simplest method is to install TensorFlow using the :ref:`Open-CE Conda distribution<software-applications-open-ce>`.


Installing via Conda (Open-CE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With a working Conda installation (see :ref:`Installing Miniconda<software-applications-conda-installing>`) the following instructions can be used to create a Python 3.8 conda environment named ``tf-env`` with the latest Open-CE provided TensorFlow:

.. note:: 

   TensorFlow installations via conda can be relatively large. Consider installing your miniconda (and therfore your conda environments) to the ``/nobackup`` file store.


.. code-block:: bash

   # Create a new conda environment named tf-env within your conda installation
   conda create -y --name tf-env python=3.8

   # Activate the conda environment
   conda activate tf-env

   # Add the OSU Open-CE conda channel to the current environment config
   conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

   # Also use strict channel priority
   conda config --env --set channel_priority strict

   # Install the latest available version of Tensorflow
   conda install -y tensorflow

In subsequent interactive sessions, and when submitting batch jobs which use TensorFlow, you will then need to re-activate the conda environment.

For example, to verify that TensorFlow is available and print the version:

.. code-block:: bash

   # Activate the conda environment
   conda activate tf-env

   # Invoke python
   python3 -c "import tensorflow;print(tensorflow.__version__)"

.. note::
   
   The :ref:`Open-CE<software-applications-open-ce>` distribution of TensorFlow does not include IBM technologies such as DDL or LMS, which were previously available via :ref:`WMLCE<software-applications-wmlce>`. 
   WMLCE is no longer supported.

Further Information
~~~~~~~~~~~~~~~~~~~

For further information on TensorFlow features and usage, please refer to the `TensorFlow Documentation <https://www.tensorflow.org/api_docs/>`__. 