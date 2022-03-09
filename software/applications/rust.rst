.. _software-rustlang:

Rust
====

The `Rust Programming Language <https://www.rust-lang.org/>`__ is a general purpose programming language designed for performance and safety.
More information on features of the Rust programming language can be found on the `rustlang.org website <https://www.rust-lang.org/learn>`__.

On Bede, Rust is available without the need to load any software modules under RHEL 8. 

The central installation includes:

* ``rustc`` - The Rust compiler
* ``cargo`` - The Rust package manager
* ``rustdoc`` - Tool to generate documentation from Rust source code


To find the version of rust currently available, run:

.. code-block:: bash

    rustc --version

.. note::

    Rust is not centrally installed on RHEL 7 images. 


If you require a different version of ``rustc`` than provided by the RHEL 8 Bede image, it should be possible to install locally into your ``/users`` directory, the ``/project`` or ``/nobackup`` file stores to avoid filling your users directory.
These methods have not been tested on Bede.

This should be possible via :ref:`Spack<software-spack>` via the `rust spack package <https://spack.readthedocs.io/en/latest/package_list.html#rust>`__ which provides the rust programming language toolchain.

Alternatively, a ``powerpc64le`` `rustup installer <https://rust-lang.github.io/rustup/installation/other.html>`__ is available. The `rustup installation instructions <https://rust-lang.github.io/rustup/installation/index.html#choosing-where-to-install>`__ include a section on controlling the installation location, via the ``RUSTUP_HOME`` and ``CARGO_HOME`` environment variables which must be set prior to executing the installation script. 