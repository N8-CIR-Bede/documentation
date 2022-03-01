.. _software-compilers-nvcc:

CUDA and NVCC
=============

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

``nvcc`` is also available within the :ref:`NVIDIA HPC SDK<software-compilers-nvhpc>` compiler tool chain.

``nvcc`` supports a wide range of command options, a full list of which can be found in the `NVCC Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options>`__.

Of these options, the following are likely to be important for Bede users.


Setting the C++ language dialect
--------------------------------

The C++ dialect used for host and device code can be controlled using the ``--std`` or ``-std`` option (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-std>`__)
, which accepts values of:

* ``c++03``
* ``c++11``
* ``c++14``

CUDA ``>= 11.0`` also accepts

* ``c++17``

The default C++ dialect depends on the host compiler, with ``nvcc`` matching the default dialect by the host c++ compiler.

.. code-block:: bash

   nvcc --std=c++14 -o main main.cu

Optimisation Level and Debug Symbols
------------------------------------

NVCC supports specifying the optimisation level of host (CPU) code via the ``--optimize`` / ``-O`` option (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-optimize>`__). 

For example, to compile using level 2 host optimisations:

.. code-block:: bash

   nvcc -O2 -o main main.cu

The optimisation level for Device code must be forwarded to the ``ptxas`` compilation phase via it's ``--opt-level`` / ``-O`` option (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-opt-level>`__)
, which defaults to a value of ``3``.

For example to compile with level 2 optimisations for host and device code

.. code-block:: bash

   nvcc -O2 -Xptxas -O2 -o main main.cu

Debug symbols can be enabled for host code and device code independently.

Host debug symbols can be enabled via ``--debug`` or ``-g`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-debug>`__). 

.. code-block:: bash

   nvcc -g -o main main.cu

Device debug symbols can be enabled via ``--device-debug`` or ``-G`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-device-debug>`__). 
Enabling device debug symbols will disable all device optimisations, resulting in significantly increased run times. 

To build an executable with debug symbols for host and device code with optimisations disabled:

.. code-block:: bash

   nvcc -g -G -O0 -o main main.cu

To enhance profiling of device code with debug symbols, use ``--generate-line-info`` or ``-lineinfo`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-generate-line-info>`__).
``-lineinfo`` and ``-G`` are mutually exclusive for recent CUDA versions. 

.. code-block:: bash

   nvcc -O3 -lineinfo -o main main.cu

GPU Code Generation Options
---------------------------

The ``-gencode`` or ``arch`` and ``-code`` NVCC compiler options allow for architecture specific optimisation of generated code, for NVCC's `two-stage compilation process <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures>`__.

Bede contains NVIDIA Tesla V100 and Tesla T4 GPUs, which are `compute capability <https://developer.nvidia.com/cuda-gpus>`__ ``7.0`` and ``7.5`` respectively.

To generate optimised code for both GPU models in Bede, the following ``-gencode`` options can be passed to ``nvcc``:

.. code-block:: bash

   nvcc -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -o main main.cu

Alternatively, to reduce compile time and binary size a single ``-gencode`` option can be passed. 

If only compute capability ``70`` is selected, code will be optimised for Volta GPUs, but will execute on Volta and Turing GPUs.

If only compute capability ``75`` is selected, code will be optimised for Turing GPUs, but it will not be executable on Volta GPUs.

.. code-block:: bash

   # Optimise for V100 GPUs, executable on T4 GPUs
   nvcc -gencode=arch=compute_70,code=sm_70 -o main main.cu
   # Optimise for T4 GPUs, not executable on V100 GPUs
   nvcc -gencode=arch=compute_75,code=sm_75 -o main main.cu

For more information on the use of ``-gencode``, ``-arch`` and ``-code`` please  see the `NVCC Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`__.


Setting Host Compiler Options
-----------------------------

By default, NVCC will error if it encounters any unknown compiler options, such as ``-march=native``, which are intended for the host compiler or linker.

This can be resolved either by instructing ``nvcc`` to forward unknown options to the host compiler and/or linker, or by explicitly passing the options to the appropriate compilation phase.

To forward unknown options to the host compiler, use ``--forward-unknown-to-host-compiler`` / ``-forward-unknown-to-host-compiler`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-forward-host-compiler>`__).

E.g. to pass ``-march=native`` and ``-Wall`` to the host compiler:

.. code-block:: bash

   nvcc --forward-unknown-to-host-compiler -march=native -Wall -o main main.cu

To forward unknown options to the host linker, use ``--forward-unknown-to-host-linker`` / ``-forward-unknown-to-host-linker``.

(`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-forward-host-linker>`__)

To forward specific options to the various compilation tools encapsulated within ``nvcc`` the following options may be used (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options>`__):

* ``--compiler-options`` / ``-Xcompiler`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-compiler-options>`__)

  * Forwards options to the compiler / preprocessor

* ``--linker-options`` / ``-Xlinker`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-linker-options>`__)

  * Options for the host linker

* ``--archive-options`` / ``-Xarchive`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-archive-options>`__)

  * Options for the library manager

* ``--ptxas-options`` / ``-Xptxas`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-ptxas-options>`__)

  * Options for the PTX optimizing assembler (``ptxas``)

* ``--nvlink-options`` / ``-Xnvlink`` (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-nvlink-options>`__)

  * Options for the device linker (``nvlink``)

When specifying options for one of the encapsulated tools, you can pass multiple options at once, separated by commas without spaces, or by enclosing multiple options passed to ``Xcompiler`` etc with double quotes. 

E.g. to pass ``-march=native`` and ``-Wall`` to the host compiler:

.. code-block:: bash

   # Pass multiple arguments using multiple -Xcompiler switches
   nvcc -Xcompiler -march=native -Xcompiler -Wall -o main main.cu

   # Pass multiple arguments separated by commas with no spaced
   nvcc -Xcompiler -march=native,-Wall -o main main.cu

   # Use double quotes to encapsulate multiple space separated options
   nvcc -Xcompiler "-march=native -Wall" -o main main.cu


Host Compiler Selection
-----------------------

``nvcc`` requires a general purpose C++ host compiler during CUDA compilation, and assumes that the host compiler has been installed using the tools default options.

By default, ``nvcc`` will use the default host compiler (``gcc`` and ``g++`` under linux) found in current execution search paths, unless specified using compiler options.

I.e. on Bede, the actively loaded ``gcc`` or ``g++`` module (see :ref:`GCC<software-compilers-gcc>` for more information).

The automatic use of ``gcc`` / ``g++`` from the path may be overridden using the ``--compiler-bindir`` / ``-ccbin`` options (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#file-and-path-specifications-compiler-bindir>`__).

This option can be used to specify the directory in which the host compiler resides, and optionally may include the binary name itself, if for instance you wish to use ``clang++`` or ``xl`` as your host C++ compiler. 

e.g. to use ``xlc++`` as the host compiler for the default CUDA module:

.. code-block:: bash

   module load xl # RHEL 8 only
   module load cuda

   nvcc -ccbin $(which xlc++) --std=c++11 -o main main.cu

``nvcc`` does check for host compiler compatibility against known compiler versions, and may error if a compiler is too new, too old or generally unknown.
This behaviour can be prevented using the ``--allow-unsupported-compiler`` / ``-allow-unsupported-compiler`` option (`docs <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#file-and-path-specifications-allow-unsupported-compiler>`__), however, this may result in incorrect binaries. Use at your own risk.

A list of officially supported host compilers can be found in the `CUDA Installation Guide for Linux <https://docs.nvidia.com/cuda/archive/11.5.2/cuda-installation-guide-linux/index.html>`__, for the appropriate CUDA version.
For Bede, refer to the Power 9 section of the table with RHEL for the operating system.