NVIDIA Profiling Tools
======================

Nvidia provide a suite of profiling tools which can be used to profile applications running on the Volta and Turing architecture Nvidia GPUs within Bede. 

`Nsight Systems <https://developer.nvidia.com/nsight-systems>`__ and `Nsight Compute <https://developer.nvidia.com/nsight-compute>`__ are the modern profiling tools introduced with CUDA 10.0, and available for use on Bede.
The `NVIDIA Visual Profiler <https://developer.nvidia.com/nvidia-visual-profiler>`_ is the legacy Nvidia profiling tool. It is recommended to use the newer tools where possible.

.. note::

   The GUI for these tools are not available on bede for use with X forwarding, as they are not available on PPC64LE.
   Profile data must be generated using the command line interfaces, which can then be explored using a local installation of the appropriate tool, which can be installed locally without requiring a local NVIDIA GPU.

Preparing your Application
--------------------------

To improve the effectiveness of the Nvidia profiling tools, several steps can be taken.

Applications compiled with ``nvcc`` should pass ``-lineinfo`` (or ``--generate-line-info``) to embed line-level profile information in the generated binary files (for Nsight Compute).

Applications compiled with the `NVIDIA HPC SDK
<https://developer.nvidia.com/hpc-sdk>`__ family of compilers should use ``-gpu=lineinfo`` to embed line-level information for use in profiling.

The :ref:` NVIDIA Tools Extension` can be used to mark regions of code. This can improve usability of the timeline view, or can be used to allow more specific profiling of applications.


Nsight Systems
--------------

Nsight Systems is a system-wide performance analysis tool designed to visualize an application’s algorithms and identify the largest opportunities to optimize.
It supports Pascal (SM 60) and newer GPUs.

A common use-case for Nsight Systems is to generate application timelines via the command line, which can later be visualised on a local computer using the GUI component.

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


Cluster Modules
~~~~~~~~~~~~~~~

``nsys`` is available through the following Bede modules:

* ``nsight-systems/2020.3.1``
* ``nvhpc/20.9``

More Information
~~~~~~~~~~~~~~~~

* `Nsight Systems <https://docs.nvidia.com/nsight-systems/>`_
* `OLCF: Nsight Systems Tutorial <https://vimeo.com/398838139>`_
  
  * Use the following `Nsight report files <https://drive.google.com/open?id=133a90SIupysHfbO3mlyfXfaEivCyV1EP>`_ to follow the tutorial.

Nsight Compute
--------------

Nsight Compute is a kernel profiler for CUDA applications, which can also be used for API debugging.
It supports Volta architecture GPUs and newer (SM 70+).

A common use-case for using Nsight Compute on HPC systems is to capture all available profiling metrics for a run of a target application, storing the information to a file on disk. This file can then be interrogated on a local machine using the Nsight Compute GUI.

For example, the following command captures the full set of metrics for an application using using the command line tool `ncu`.

.. code-block:: bash

   ncu -o profile --set full ./myapplication <arguments>


Capturing the full set of metrics can lead to very long run times, as each kernel is replayed many times.
Rather than capturing the full set of metrics, a subset may be captured using the ``--set``, ``--section`` and ``--metrics`` flags as described in the `Nsight Comptue Profile Command Line Options table <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile>`_.

The scope of the section being profiled can also be reduced using `NVTX Filtering <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering>`_; or by targetting specific kernels using ``--kernel-id``, ``--kernel-regex`` and/or ``--launch-skip`` see the `CLI docs for more information <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile>`_).


Once the ``.ncu-rep`` file has been downloaded locally, it can be imported into local Nsight CUDA GUI ``ncu-ui`` via ``ncu-ui profile.ncu-rep`` **or**  ``File > Open > profile.ncu-rep`` in the GUI.

.. note::
   Older versions of Nsight Compute (CUDA < v11.0.194) provided ``nv-nsight-cu-cli`` ``nv-nsight-cu`` rather than ``ncu`` and ``ncu-ui`` respectively.

   The generated report file used the ``.nsight-cuprof-report`` extension rather than ``.ncu-rep``.


Cluster Modules
~~~~~~~~~~~~~~~

``ncu`` is available through the following Bede modules:

* ``nsight-compute/2020.2.1``
* ``nvhpc/20.9``


More Information
~~~~~~~~~~~~~~~~

* `Nsight Compute <https://docs.nvidia.com/nsight-compute/>`_
* `OLCF: Nsight Compute Tutorial <https://vimeo.com/398929189>`_

  * Use the following `Nsight report files <https://drive.google.com/open?id=133a90SIupysHfbO3mlyfXfaEivCyV1EP>`_ to follow the tutorial.


Nvidia Visual Profiler (legacy)
-------------------------------

The Visual Profiler is NVIDIA's legacy profiler, which fills some of the roles of bother Nsight Systems and Nsight Compute, but is no longer actively developed.
It is still provided to enable profiling of older GPU architectures not supported by the newer tools.
All features are supported by the Volta architecture GPUs in Bede, but kernel profiling is **not** supported for the Turing architecture GPUs.
It is recommended to use the newer Nsight Systems and Nsight Compute tools.


Application timelines can be generated using ``nvprof``:

.. code-block:: bash

   nvprof -o timeline.nvprof ./myapplication <arguments>


Fine-grained kernel profile information can be genereted remotely using ``nvprof``:

.. code-block:: bash

   nvprof --analysis-metrics -o analysis.nvprof ./myapplication <arguments>

This captures the full set of metrics required to complete the guided analysis, and may take a (very long) while.
Large applications request fewer metrics (via ``--metrics``), fewer events (via ``--events``) or target specific kernels (via ``--kernels``). See the `nvprof command line options <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_ for further information.

Once these files are downloaded to your local machine, Import them into the Visual Profiler GUI (``nvvp``)

* ``File > Import``
* Select ``Nvprof``
* Select ``Single process``
* Select ``timeline.nvvp`` for ``Timeline data file``
* Add ``analysis.nvprof`` to ``Event/Metric data files``

Cluster Modules
~~~~~~~~~~~~~~~

``nvprof`` is available through the following Bede modules:

* ``cuda/10.1.243``
* ``cuda/10.2.89``
* ``nvhpc/20.9``

Documentation
~~~~~~~~~~~~~

+ `Nvprof Documentation <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_


NVIDIA Tools Extension
----------------------

`NVIDIA Tools Extension (NVTX) <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`__ is a C-based API for annotating events and ranges in applications.
These markers and ranges can be used to increase the usability of the NVIDIA profiling tools.


* For CUDA ``>= 10.0``, NVTX version ``3`` is distributed as a header only library.
* For CUDA ``<  10.0``, NVTX is distributed as a shared library.

The location of the headers and shared libraries may vary between Operating Systems, and CUDA installation (i.e. CUDA toolkit, PGI compilers or HPC SDK).

The NVIDIA Developer blog contains several posts on using NVTX:

* `Generate Custom Application Profile Timelines with NVTX (Jiri Kraus) <https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/>`_
* `Track MPI Calls In The NVIDIA Visual Profiler (Jeff Larkin) <https://developer.nvidia.com/blog/gpu-pro-tip-track-mpi-calls-nvidia-visual-profiler/>`_
* `Customize CUDA Fortran Profiling with NVTX (Massimiliano Fatica) <https://developer.nvidia.com/blog/customize-cuda-fortran-profiling-nvtx/>`_


CMake support
~~~~~~~~~~~~~

From CMake 3.17, the `FindCUDAToolkit module <https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html>`_ can be used to find the tools extension and select the appropriate include directory.

If support for older CMake versions is required custom ``find_package`` modules can be used, e.g. `ptheywood/cuda-cmake-NVTX on GitHub <https://github.com/ptheywood/cuda-cmake-nvtx>`_.


Documentation
~~~~~~~~~~~~~

* `NVTX Documentation <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_
* `NVTX 3 on GitHub <https://github.com/NVIDIA/NVTX>`_
