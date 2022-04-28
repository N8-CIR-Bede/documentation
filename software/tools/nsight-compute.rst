.. _software-tools-nsight-compute:

Nsight Compute
==============

`Nsight Compute <https://developer.nvidia.com/nsight-compute>`__ is a kernel profiler for CUDA applications, which can also be used for API debugging.
It supports Volta architecture GPUs and newer (SM 70+).

On Bede, Nsight Compute is provided by a number of modules, with differing versions of ``ncu``.
The GUI is not available on Bede, but profile data can be generated on Bede via the CLI.

You should use a versions of ``ncu`` that is at least as new as the CUDA toolkit used to compile your application (if appropriate).

.. code-block:: bash

   module load nsight-compute/2022.1.0 # provides nsys 2022.1.0
   module load nsight-compute/2020.2.1 # provides nsys 2020.2.1

   # RHEL 7 only
   module load nvhpc/20.9  # provides nsys 2020.1.0

   # RHEL 8 only
   module load cuda/11.5.1 # provides nsys 2021.3.1
   module load cuda/11.4.1 # provides nsys 2021.2.1
   module load cuda/11.3.1 # provides nsys 2021.1.1
   module load cuda/11.2.2 # provides nsys 2020.3.1
   module load nvhpc/22.1  # provides nsys 2021.3.0
   module load nvhpc/21.5  # provides nsys 2021.1.0


Consider compiling your CUDA application using ``nvcc`` with ``-lineinfo`` or ``--generate-line-info`` to generate line-level profile information.

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

More Information
^^^^^^^^^^^^^^^^

* `Nsight Compute <https://docs.nvidia.com/nsight-compute/>`_
* `OLCF: Nsight Compute Tutorial <https://vimeo.com/398929189>`_

  * Use the following `Nsight report files <https://drive.google.com/open?id=133a90SIupysHfbO3mlyfXfaEivCyV1EP>`_ to follow the tutorial.