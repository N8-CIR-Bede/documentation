.. _software-applications-pytorch:

PyTorch
-------

`PyTorch <https://pytorch.org/>`__ is an end-to-end machine learning framework.
PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

The main method of distribution for PyTorch is via :ref:`Conda <software-applications-conda>`.

For more information on the usage of PyTorch, see the `Online Documentation <https://pytorch.org/docs/>`__.

PyTorch Quickstart
~~~~~~~~~~~~~~~~~~

The following should get you set up with a working conda environment (replacing <project> with your project code):

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
    conda config --set channel_priority strict

    conda config --prepend channels \
            https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

    conda config --prepend channels \
            https://opence.mit.edu

    conda create --name opence pytorch=1.7.1 -y
    conda activate opence


This has some limitations such as not supporting large model support. 
If you require LMS, please see the :ref:`WMLCE <software-applications-wmlce>` page.


Further Information
~~~~~~~~~~~~~~~~~~~

For more information on the usage of PyTorch, see the `Online Documentation <https://pytorch.org/docs/>`__.