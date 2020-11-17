.. _hardware:

Hardware
--------

The system is based around the IBM POWER9 CPU and NVIDIA Tesla GPUs.
Connectivity within a node is optimised by both the CPUs and GPUs being
connected to an NVIDIA NVLink 2.0 bus, and outside of a node by a
dual-rail Mellanox EDR InfiniBand interconnect allowing GPUDirect RDMA
communications (direct memory transfers to/from GPU memory).

Together with IBM’s software engineering, the POWER9 architecture is
uniquely positioned for:

-  Large memory GPU use, as the GPUs are able to access main system
   memory via POWER9’s large model feature.
-  Multi node GPU use, via IBM’s Distributed Deep Learning (DDL)
   software.

There are:

-  2x “login” nodes, each containing:

   -  2x POWER9 CPUs @ 2.4GHz (40 cores total and 4 hardware threads per
      core), with NVLink 2.0
   -  512GB DDR4 RAM
   -  4x Tesla V100 32G NVLink 2.0
   -  1x Mellanox EDR (100Gbit/s) InfiniBand port

-  32x “gpu” nodes, each containing:

   -  2x POWER9 CPUs @ 2.7GHz (32 cores total and 4 hardware threads per
      core), with NVLink 2.0
   -  512GB DDR4 RAM
   -  4x Tesla V100 32G NVLink 2.0
   -  2x Mellanox EDR (100Gbit/s) InfiniBand ports

-  4x “infer” nodes, each containing:

   -  2x POWER9 CPUs @ 2.9GHz (40 cores total and 4 hardware threads per
      core)
   -  256GB DDR4 RAM
   -  4x Tesla T4 16G PCIe
   -  1x Mellanox EDR (100Gbit/s) InfiniBand port

The Mellanox EDR InfiniBand interconnect is organised in a 2:1 block fat
tree topology. GPUDirect RDMA transfers are supported on the 32 “gpu”
nodes only, as this requires an InfiniBand port per POWER9 CPU socket.

Storage is provided by a 2PB Lustre filesystem capable of reaching
10GB/s read or write performance, supplemented by an NFS service
providing modest home and project directory needs.
