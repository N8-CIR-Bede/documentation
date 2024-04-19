.. _software-libraries-nvtoolsext:

NVIDIA Tools Extension
~~~~~~~~~~~~~~~~~~~~~~

`NVIDIA Tools Extension (NVTX) <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`__ is a C-based API for annotating events and ranges in applications.
These markers and ranges can be used to increase the usability of the NVIDIA profiling tools.

* For CUDA ``>= 10.0``, NVTX version ``3`` is distributed as a header only library.
* For CUDA ``<  10.0``, NVTX is distributed as a shared library.

The location of the headers and shared libraries may vary between Operating Systems, and CUDA installation (i.e. CUDA toolkit, PGI compilers or HPC SDK).

On Bede, ``nvToolsExt`` is provided by the :ref:`CUDA <software-compilers-nvcc>` and :ref:`NVHPC <software-compilers-nvhpc>` modules:

.. code-block:: bash
    
   module load cuda
   module load nvhpc

The NVIDIA Developer blog contains several posts on using NVTX:

* `Generate Custom Application Profile Timelines with NVTX (Jiri Kraus) <https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/>`_
* `Track MPI Calls In The NVIDIA Visual Profiler (Jeff Larkin) <https://developer.nvidia.com/blog/gpu-pro-tip-track-mpi-calls-nvidia-visual-profiler/>`_
* `Customize CUDA Fortran Profiling with NVTX (Massimiliano Fatica) <https://developer.nvidia.com/blog/customize-cuda-fortran-profiling-nvtx/>`_


CMake support
^^^^^^^^^^^^^

From CMake 3.17, the ```FindCUDAToolkit <https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html>`_`` can be used to find the tools extension and select the appropriate include directory.
If support for older CMake versions is required custom ``find_package`` modules can be used, e.g. `ptheywood/cuda-cmake-NVTX on GitHub <https://github.com/ptheywood/cuda-cmake-nvtx>`_.


Documentation
^^^^^^^^^^^^^

* `NVTX Documentation <https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_
* `NVTX 3 on GitHub <https://github.com/NVIDIA/NVTX>`_
