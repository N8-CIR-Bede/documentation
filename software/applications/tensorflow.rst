.. _software-python-tensorflow:

TensorFlow
----------

`TensorFlow <https://www.tensorflow.org/>`__ is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

TensorFlow Quickstart
~~~~~~~~~~~~~~~~~~~~~

The following should get you set up with a working conda environment (replacing ``<project>`` with your project code):

.. code-block:: bash

    export DIR=/nobackup/projects/<project>/$USER
    # rm -rf ~/.conda ~/.condarc $DIR/miniconda # Uncomment if you want to remove old env
    mkdir $DIR
    pushd $DIR

    # Download the latest miniconda installer for ppcle64
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
    # Validate the file checksum matches is listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
    sha256sum Miniconda3-latest-Linux-ppc64le.sh

    sh Miniconda3-latest-Linux-ppc64le.sh -b -p $DIR/miniconda
    source miniconda/bin/activate
    conda update conda -y

    conda config --prepend channels \
            https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

    conda config --prepend channels \
            https://opence.mit.edu

    conda create --name opence tensorflow -y
    conda activate opence

.. note::
  
   This conflicts with the :ref:`PyTorch <software-applications-pytorch>` instructions as they set the conda channel_priority to be strict which seems to cause issues when installing TensorFlow.

Further Information
~~~~~~~~~~~~~~~~~~~

For further information on TensorFlow features and usage, please refer to the `TensorFlow Documentation <https://www.tensorflow.org/api_docs/>`__. 