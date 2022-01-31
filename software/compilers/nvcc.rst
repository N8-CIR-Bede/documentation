.. _software-compilers-nvcc:

CUDA and NVCC
-------------

`CUDA <https://developer.nvidia.com/cuda-zone>`__ and the ``nvcc`` CUDA/C++ compiler are provided for use on the system by the `cuda` modules.

Unlike other compiler modules, the cuda modules do not set ``CC`` or ``CXX`` environment variables. This is because ``nvcc`` can be used to compile device CUDA code in conjunction with a range of host compilers, such as GCC or LLVM clang.


.. code-block:: bash

   module load cuda

   # RHEL 8 only
   module load cuda/11.5.1
   module load cuda/11.4.1
   module load cuda/11.3.1
   module load cuda/11.2.2

   # RHEL 7 or RHEL 8
   module load cuda/10.2.89
   module load cuda/10.1.243

For further information please see the `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-archive>`__.


GPU Code Generation Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``-gencode`` or ``arch`` and ``-code`` NVCC compiler options allow for architecture specific optimisation of generated code, for NVCC's `two-stage compilation process <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures>`__.

Bede contains NVIDIA Tesla V100 and Tesla T4 GPUs, which are `compute capability <https://developer.nvidia.com/cuda-gpus>`__ ``7.0`` and ``7.5`` respectively.

To generate optimised code for both GPU models in Bede, the following ``-gencode`` options can be passed to ``nvcc``:

.. code-block:: bash

   -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75

Alternatively, to reduce compile time and binary size a single ``-gencode`` option can be passed. 

If only compute capability ``70`` is selected, code will be optimised for Volta GPUs, but will execute on Volta and Turing GPUs.

If only compute capability ``75`` is selected, code will be optimised for Turing GPUs, but it will not be executable on Volta GPUs.

.. code-block:: bash

   # Optimise for V100 GPUs, executable on T4 GPUs
   -gencode=arch=compute_70,code=sm_70 
   # Optimise for T4 GPUs, not executable on V100 GPUs
   -gencode=arch=compute_75,code=sm_75

For more information on the use of ``-gencode``, ``-arch`` and ``-code`` please  see the `NVCC Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`__.