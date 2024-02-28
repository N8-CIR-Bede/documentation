.. _software-tools-nsight-systems:

Nsight Systems
==============

`Nsight Systems <https://developer.nvidia.com/nsight-systems>`__ is a system-wide performance analysis tool designed to visualize an application’s algorithms and identify the largest opportunities to optimize.
It supports Pascal (SM 60) and newer GPUs.

A common use-case for Nsight Systems is to generate application timelines via the command line, which can later be visualised on a local computer using the GUI component.
The GUI is not available on Bede.

On Bede, Nsight Systems is provided by a number of modules, with differing versions of ``nsys``. 
You should use a versions of ``nsys`` that is at least as new as the CUDA toolkit used to compile your application (if appropriate).

.. code-block:: bash

   module load nsight-systems/2022.1.1 # provides nsys 2022.1.1
   module load nsight-systems/2020.3.1 # provides nsys 2020.3.1

   # RHEL 7 only
   module load nvhpc/20.9  # provides nsys 2020.3.1

   # RHEL 8 only
   module load cuda/11.5.1 # provides nsys 2021.3.3
   module load cuda/11.4.1 # provides nsys 2021.2.4
   module load cuda/11.3.1 # provides nsys 2021.1.3
   module load cuda/11.2.2 # provides nsys 2020.4.3
   module load nvhpc/22.1  # provides nsys 2021.5.1
   module load nvhpc/21.5  # provides nsys 2021.2.1

To generate an application timeline with Nsight Systems CLI (``nsys``):

.. code-block:: bash

   nsys profile -o timeline ./myapplication <arguments>

Nsight systems can trace mulitple APIs, such as CUDA and OpenACC. 
The ``--trace`` argument to specify which APIs should be traced.
See the `nsys profiling command switch options <https://docs.nvidia.com/nsight-systems/profiling/index.html#cli-profile-command-switch-options>`__ for further information.

.. code-block:: bash

   nsys profile -o timeline --trace cuda,nvtx,osrt,openacc ./myapplication <arguments>

.. note::
   On Power9 systems such as Bede the ``--trace`` option ``osrt`` can lead to ``SIGILL`` errors with some versions of ``nsys``. As this is part of the default set default, consider passing ``--trace cuda,nvtx`` instead.

Once this file has been downloaded to your local machine, it can be opened in ``nsys-ui``/``nsight-sys`` via ``File > Open > timeline.qdrep``


More Information
^^^^^^^^^^^^^^^^

* `Nsight Systems <https://docs.nvidia.com/nsight-systems/>`_
* `OLCF: Nsight Systems Tutorial <https://vimeo.com/398838139>`_
  
  * Use the following `Nsight report files <https://drive.google.com/open?id=133a90SIupysHfbO3mlyfXfaEivCyV1EP>`_ to follow the tutorial.