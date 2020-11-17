.. _bede_hardware:

Bede Hardware
=============

* 32x Main GPU nodes, each node (IBM AC922) has:
    * 512GB DDR4 RAM
    * 2x IBM POWER9 CPUs (and two NUMA nodes), with
    * 4x NVIDIA V100 GPUs (2 per CPU)
    * Each CPU is connected to its two GPUs via high-bandwidth, low-latency NVLink interconnects (helps if you need to move lots of data to/from GPU memory)
* 4x Inferencing Nodes:
    * Equipped with T4 GPUs for inference tasks.
* 2x Visualisation nodes
* 2x Login nodes
* 100 Gb EDR Infiniband (high bandwith and low latency to support multi-node jobs)
* 2PB Lustre parallel file system (available over Infiniband and Ethernet network interfaces)