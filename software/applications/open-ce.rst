.. _software-applications-open-ce:

Open-CE
=======

.. |arch_availabilty_name| replace:: Open-CE
.. include:: /common/ppc64le-only-sidebar.rst

The `Open Cognitive Environment (Open-CE) <https://osuosl.org/services/powerdev/opence/>`__ is a community driven software distribution for machine learning and deep learning frameworks.

Open-CE software is distributed via :ref:`Conda<software-applications-conda>`, with all included packages for a given Open-CE release being installable in to the same conda environment.

Open-CE conda channels suitable for use on Bede's IBM Power architecture systems are hosted by `Oregon State University <https://osuosl.org/services/powerdev/opence/>`__ and `MIT <https://opence.mit.edu/>`__.

It is the successor to :ref:`IBM WMLCE <software-applications-wmlce>` which was archived on 2020-11-10, with IBM WMLCE 1.7.0 being the final release.

Open-CE includes the following software packages, amongst others:

* :ref:`TensorFlow <software-applications-tensorflow>`
* :ref:`PyTorch <software-applications-pytorch>`
* `Horovod <https://horovod.ai/>`__
* `ONNX <https://onnx.ai/>`__

.. note:: 

   Open-CE does not include all features from WMLCE, such as Large Model Support or Distributed Deep Learning (DDL). 

Using Open-CE
-------------

Open-CE provides software packages via :ref:`Conda<software-applications-conda>`, which you must first :ref:`install<software-applications-conda-installing>`.
Conda installations of the packages provided by Open-CE can become quite large (multiple GBs), so you may wish to use a conda installation in ``/nobackup/projects/<project>`` or ``/projects/<project>`` as described in the :ref:`Installing Conda section <software-applications-conda-installing>`.

With a working Conda install, Open-CE packages can be installed from either the OSU or MIT Conda channels for PPC64LE systems such as Bede.

* OSU: ``https://ftp.osuosl.org/pub/open-ce/current/``
* MIT: ``https://opence.mit.edu/``

Using Conda Environments are recommended when working with Open-CE.

I.e. to install ``tensorflow`` and ``pytorch`` from OSU Open-CE conda channel into a conda environment named ``open-ce``:

.. code-block:: bash

   # Create a new conda environment named open-ce within your conda installation
   conda create -y --name open-ce python=3.9 # Older Open-CE may require older Python versions

   # Activate the conda environment
   conda activate open-ce

   # Add the OSU Open-CE conda channel to the current environment config
   conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/
   # Also use strict channel priority
   conda config --env --set channel_priority strict

   # Install the required conda package, using the channels set within the conda env. This may take some time.
   conda install -y tensorflow
   conda install -y pytorch

Once installed into a conda environment, the Open-CE provided software packages can be used interactively on login nodes or within batch jobs by activating the named conda environment.

.. code-block:: bash

   # Activate the conda environment
   conda activate open-ce

   # Run a python command or script which makes use of the installed packages
   # I.e. to output the version of tensorflow:
   python3 -c "import tensorflow;print(tensorflow.__version__)"

   # I.e. or to output the version of pytorch:
   python3 -c "import torch;print(torch.__version__)"

Using older versions of Open-CE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OSU conda distribution provides an archive of older Open-CE releases, beginning at version ``1.0.0``. 

The available versions are listed at https://ftp.osuosl.org/pub/open-ce/.

Using versions other than ``current`` can be done by modifying the channel URI when adding the channel to the current conda environment with the desired version number. 

I.e. to explicitly use Open-CE ``1.4.1`` the command to add the conda channel to the current environment would be:

.. code-block:: bash

   conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/1.4.1/

Using older Open-CE versions may require older python versions. 
See the `OSU Open-CE page <https://osuosl.org/services/powerdev/opence/>`__ for further version information.

The MIT Open-CE channel provides multiple versions of Open-CE in the same Conda channel. If using the MIT Open-CE distribution, older versions of packages can be requested by specifying the specific version of the desired package.

Why use Open-CE
---------------

Modern machine learning packages like TensorFlow and PyTorch have large dependency trees which can conflict with one another due to the independent release schedules.
This has made it difficult to use multiple competing packages within the same environment. 

Open-CE solves this issue by ensuring that packages included in a given Open-CE distribution are compatible with one another, and can be installed a the same time, simplifying the distribution of these packages. 

It also provides pre-compiled distributions of these packages for PPC64LE architecture machines, which are not always available from upstream sources, reducing the time required to install these packages.

For more information on the potential benefits of using Open-CE see `this blog post from the OpenPOWER foundation <https://openpowerfoundation.org/blog/open-cognitive-environment-open-ce-a-valuable-tool-for-ai-researchers/>`__.

Differences from WMLCE
----------------------

:ref:`IBM WMLCE<software-applications-wmlce>` include several features not available in upstream TensorFlow and PyTorch distributions, such as Large Model Support.

Unfortunately, LMS is not available in TensorFlow or PyTorch provided by Open-CE.

Other features or packages absent in Open-CE which were included in WMLCE include:

* Large Model Support (LMS)
* IBM DDL
* Caffe (IMB-enhanced)
* IBM SnapML
* NVIDIA Rapids

