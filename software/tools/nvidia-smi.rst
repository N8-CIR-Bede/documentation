.. _software-tools-nvidia-smi:

NVIDIA-SMI
===========

``nvidia-smi`` is the NVIDIA System Management Interface.
It is a command line tool which provides monitoring information for NVIDIA GPUs.

It is available for use by default in interactive and batch sessions on Bede, but operations which would require root will not be usable by regular Bede users.

Most Bede users will not need to interact with ``nvidia-smi``, however, it can be used to gather information about GPUs in a system and how they are connected to one another, which may be useful when reporting any performance results.

Using ``nvidia-smi``
--------------------

Running the ``nvidia-smi`` tool without any arguments will present summary information about the available GPUs on the current node that are accessible by the user, and information about the GPU driver in use. 

.. code-block:: bash

   nvidia-smi

Detailed information per GPU can be queried using the ``-q`` and ``-i`` options:

.. code-block:: bash

    # View detailed information about device 0
    nvidia-smi -i 0 -q

V100 GPUs within ``gpu`` nodes in Bede are connected to one another and the CPU via NVLink connections, while T4 GPUs in ``infer`` nodes are not.
How GPUs within a node are connected to one another can be queried via the ``topo`` subcommand. 
This may be useful when using multi-GPU applications. 

.. code-block:: bash

   # View the GPUDirect communication matrix via -m / --matrix
   nvidia-smi topo -m
   # View how GPUs 0 and 1 are connected, in a session with >= 2 GPUs
   nvidia-smi topo  -i 0,1 -p

The ``nvlink`` subcommand can be used to query the status of each NVlink connection:

.. code-block:: bash

   # View the status of each nvlink for device 0
   nvidia-smi nvlink -i 0 -s
   # View how GPUs 0 and 1 are connected, in a session with >= 2 GPUs
   nvidia-smi topo  -i 0,1 -p



Full usage documentation can be found via the ``--help`` option:


.. code-block:: bash

   nvidia-smi --help