Profiling
=========

NVIDIA Profiling Tools
----------------------

HPC systems typically favour batch jobs rather than interactive jobs for improved utilsation of resources.
The Nvidia profiling tools can all be used to capture all required via the command line, which can then be interrogated using the GUI tools locally.

Nsight Systems and Nsight Compute are the modern Nvidia profiling tools, introduced with CUDA 10.0 supporting Pascal+ and Volta+ respectivley.

The NVIDIA Visual Profiler is the legacy profiling tool, with full support for GPUs up to pascal (SM < 75), partial support for Turing (SM 75 and no support for Ampere (SM80).


Compiler settings for profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applications compiled with ``nvcc`` should pass ``-lineinfo`` (or ``--generate-line-info``) to include source-level profile information.

Additionally, `NVIDIA Tools Extension SDK <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_ can be used to enhance these profiling tools.


Nsight Systems and Nsight Compute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    * Nsight Systems supports Pascal and above (SM 60+)
    * Nsight Compute supports Volta and aboce (SM 70+)

Generate an application timeline with Nsight Systems CLI (``nsys``):

.. code-block:: bash

   nsys profile -o timeline ./myapplication

Use the ``--trace`` argument to specify which APIs should be traced.
See the `nsys profiling command switch options <https://docs.nvidia.com/nsight-systems/profiling/index.html#cli-profile-command-switch-options>`_ for further information.

.. code-block:: bash

   nsys profile -o timeline --trace cuda,nvtx,osrt,openacc ./myapplication <arguments>


.. note::
   On Bede (Power9) the ``--trace`` option ``osrt`` can lead to ``SIGILL`` errors. As this is a default, consider passing ``--trace cuda,nvtx`` as an alternative minimum.


Once this file has been downloaded to your local machine, it can be opened in ``nsys-ui``/``nsight-sys`` via ``File > Open > timeline.qdrep``:


Fine-grained kernel profile information can be captured using remote Nsight Compute CLI (``ncu``/``nv-nsight-cu-cli``):

.. code-block:: bash

   ncu -o profile --set full ./myapplication <arguments>

.. note::
   ``ncu`` is available since CUDA v11.0.194, and Nsight Compute v2020.1.1. For older versions of CUDA use ``nv-nsight-cu-cli`` (if Nsight Compute is installed).


This will capture the full set of available metrics, to populate all sections of the Nsight Compute GUI, however this can lead to very long run times to capture all the information.

For long running applications, it may be favourable to capture a smaller set of metrics using the ``--set``, ``--section`` and ``--metrics`` flags as described in the `Nsight Comptue Profile Command Line Options table <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile>`_.

The scope of the section being profiled can also be reduced using `NVTX Filtering <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering>`_; or by targetting specific kernels using ``--kernel-id``, ``--kernel-regex`` and/or ``--launch-skip`` see the `CLI docs for more information <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile>`_).


Once the ``.ncu-rep`` file has been downloaded locally, it can be imported into local Nsight CUDA GUI ``ncu-ui``/``nv-nsight-cu`` via:

.. code-block:: bash

   ncu-ui profile.ncu-rep


**Or** ``File > Open > profile.ncu-rep``, **or** Drag ``profile.ncu-rep`` into the ``nv-nsight-cu`` window.

.. note::
   Older versions of Nsight Compute (CUDA < v11.0.194) used ``nv-nsight-cu`` rather than ``ncu-ui``.

.. note::
   Older versions of Nsight Compute generated ``.nsight-cuprof-report`` files, instead of ``.ncu-rep`` files.


More info
^^^^^^^^^

+ `Nsight Systems <https://docs.nvidia.com/nsight-systems/>`_
+ `Nsight Compute <https://docs.nvidia.com/nsight-compute/>`_
+ `OLCF: Nsight Systems Tutorial <https://vimeo.com/398838139>`_
+ `OLCF: Nsight Compute Tutorial <https://vimeo.com/398929189>`_

Use the following `Nsight report files <https://drive.google.com/open?id=133a90SIupysHfbO3mlyfXfaEivCyV1EP>`_ to follow the tutorial.


Cluster Modules
^^^^^^^^^^^^^^^
* ``module load nvidia/20.5``


Visual Profiler (legacy)
~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
   * Nvprof does not support CUDA kernel profiling for Turing GPUs (SM75)
   * Nvprof does not support Ampere GPUs (SM80+)

Application timelines can be generated using ``nvprof``:

.. code-block:: bash

   nvprof -o timeline.nvprof ./myapplication


Fine-grained kernel profile information can be genereted remotely using ``nvprof``:

.. code-block:: bash

   nvprof --analysis-metrics -o analysis.nvprof ./myapplication

This captuires the full set of metrics required to complete the guided analysis, and may take a (very long) while.
Large applications request fewer metrics (via ``--metrics``), fewer events (via ``--events``) or target specific kernels (via ``--kernels``). See the `nvprof command line options <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_ for further information.

Once these files are downloaded to your local machine, Import them into the Visual Profiler GUI (``nvvp``)

+ ``File > Import``
+ Select ``Nvprof``
+ Select ``Single process``
+ Select ``timeline.nvvp`` for ``Timeline data file``
+ Add ``analysis.nvprof`` to ``Event/Metric data files``


Documentation
^^^^^^^^^^^^^

+ `Nvprof Documentation <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_

Cluster Modules
^^^^^^^^^^^^^^^
   * ``module load cuda/10.1``
   * ``module load cuda/10.2``
   * ``module load nvidia/20.5``

NVIDIA Tools Extension
----------------------

NVIDIA Tools Extension (NVTX) is a C-based API for annotating events and ranges in applications.
These markers and ranges can be used to increase the usability of the NVIDIA profiling tools.


* For CUDA ``>= 10.0``, NVTX version 3 is distributed as a header only library.
* For CUDA ``<  10.0``, NVTX is distributed as a shared library.

The location of the headers and shared libraries may vary between Operating Systems, and CUDA installation (i.e. CUDA toolkit, PGI compilers or HPC SDK).



The NVIDIA Developer blog contains several posts on using NVTX:

* `Generate Custom Application Profile Timelines with NVTX (Jiri Kraus) <https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/>`_
* `Track MPI Calls In The NVIDIA Visual Profiler (Jeff Larkin) <https://developer.nvidia.com/blog/gpu-pro-tip-track-mpi-calls-nvidia-visual-profiler/>`_
* `Customize CUDA Fortran Profiling with NVTX (Massimiliano Fatica) <https://developer.nvidia.com/blog/customize-cuda-fortran-profiling-nvtx/>`_



Custom CMake ``find_package`` modules can be written to enable use within Cmake e.g. `ptheywood/cuda-cmake-NVTX on GitHub <https://github.com/ptheywood/cuda-cmake-nvtx>`_



Documentation
~~~~~~~~~~~~~

+ `NVTX Documentation <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_
+ `NVTX 3 on GitHub <https://github.com/NVIDIA/NVTX>`_
