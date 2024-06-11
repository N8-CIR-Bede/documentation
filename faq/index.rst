FAQ
=====

This page contains answers to the questions most frequently asked by Bede
users. 
The question list will be updated over time as more questions are
asked - if there is anything that you think should definitely be answered
here, please let us know by `Opening an Issue on GitHub <https://github.com/N8-CIR-Bede/documentation/issues/new>`__ or by contacting your `local Bede RSE <https://n8cir.org.uk/supporting-research/facilities/bede/rse-support-bede/>`__ .

How long can I run jobs for?
----------------------------

You can run jobs for a maximum of 48 hours, in either the ``gpu`` or ``infer``
partitions.
This is detailed in the :ref:`Usage section of the documentation<usage-maximum-job-runtime>`.

How can I acknowledge Bede in published or presented work?
----------------------------------------------------------------

You can acknowledge Bede using the standard text that we provide in :ref:`Acknowledging Bede<usage-acknowledging-bede>`:

.. include:: /common/acknowledging-bede.rst

.. _faq-ssh-fingerprints:

Login node SSH RSE/ECDSA/ED25519 host key fingerprints
------------------------------------------------------

The RSE, ECDSA and ED25519 fingerprints for Bede's login nodes are:

.. code-block::

   SHA256:w2z1RA8kRH+XFOaoHo4N610aNnItzPlKpKYLwJ5WbnY (RSA)
   SHA256:FBznAa7eKh7QdNl6P5gCiW9uy2UQ8tSGXaKJStVAU5c (ECDSA)
   SHA256:HbIkgn8V+nCLr3R99+C5uwnAWLxKTUD954qRGnzMRUM (ED25519)

How can I check my home directory quota?
----------------------------------------

You can use the following command:

.. code-block:: text

  quota -s

  Disk quotas for user XXXXXX (uid YYYYYYYY): 
       Filesystem   space   quota   limit   grace   files   quota   limit   grace
  nfs.bede.dur.ac.uk:/nfs/users
                   79544K      0K  20480M            1071       0       0

This tells me that, in this case, I have a limit of :code:`20480M`, and am 
currently using :code:`79544K` across 1071 files. 
For more information on using Bede's filesystems see the :ref:`File Storage <usage-file-storage>` documentation.

Where should I put project data?
--------------------------------

Project data that does not need to be backed up, e.g. intermediate results,
should be stored in the :code:`/nobackup/projects/<project>` directory, where
:code:`<project>` is your Bede project code.

Project data that should be backed up, e.g. final results that would be painful
to recompute, should be store in the :code:`/projects/<project>` directory.

.. note::
  Backups are **NOT** currently implemented on the :code:`/projects` filesystem!

How do I get started with Bede?
-------------------------------

The :ref:`Using Bede<using-bede>` page provides details on how to get registered, how to log in to the
machine and how to run jobs.

How can I add my own software?
------------------------------

It is recommended that you use :ref:`Spack<software-spack>` to extend the installed software on the system if possible.

Alternatively, the software you wish to use may provide instructions on local installation, or `contact your local Bede RSE <https://n8cir.org.uk/supporting-research/facilities/bede/rse-support-bede/>`__ for further guidance.

Is MATLAB available on Bede?
----------------------------

Unfortunately `MATLAB <https://www.mathworks.com/products/matlab.html>`__ is not available on Bede. MATLAB is not currently supported on IBM systems with a Power architecture.
It is possible to install `Octave <https://www.gnu.org/software/octave/index>`__ on Bede which can work as an alternative, although there is
currently no official support for this.


.. _faq-reducemfa:

How do I reduce the number of times I'm prompted for my password and a MFA code?
--------------------------------------------------------------------------------

As SSH User keys are being phased out on Bede, you may find that providing your password and an MFA code for every terminal session or file transfer can be painful. There are a number of ways to reduce the frequency of password and MFA challenges.

Windows users:

* MobaXterm.
  This program has a file transfer facility built into it. Once you login, a tree view of your home directory on Bede should be seen on the left hand side. This can be used to drag and drop files between your desktop and Bede.
* PuTTY.
  This program is able to run multiple sessions over a single SSH connection. To enable this, go to the PuTTY configuration screen and ensure the *Share SSH connections if possible* box is ticked under *Connection->SSH*.
* X2GO.
  In addition to speeding up graphical programs, X2GO allows you to launch multiple terminals within the same login session, or export a local directory so that it can be used on Bede. See :ref:`Using Bede<using-bede>` for details.

Linux/Mac OS X users:

* X2GO.
  In addition to speeding up graphical programs, X2GO allows you to launch multiple terminals within the same login session, or export a local directory so that it can be used on Bede. See :ref:`Using Bede<using-bede>` for details.
* SSH multiplexing.
  The most commonly used SSH client is called OpenSSH, which can be configured to reuse an ssh session by adding the following to your local (**not** Bede's) ``~/.ssh/config`` file:

.. code-block:: console

   Host bede login1.bede login2.bede
      CanonicalizeHostname yes
      CanonicalDomains dur.ac.uk
      ControlPath ~/.ssh/control/%C.sock
      ControlMaster auto
      ControlPersist 10m

And then running the following commands:

.. code-block:: bash

   mkdir ~/.ssh/control
   chmod 700 ~/.ssh/control

Once done, the following command will log into Bede and subsequent ssh/scp commands will reuse the connection without prompting for a password or an MFA code:

.. code-block:: bash

   ssh bede


How can I suggest improvements or contribute to this documentation?
-------------------------------------------------------------------

The Bede documentation is maintained on GitHub at
`github.com/N8-CIR-Bede/documentation <https://github.com/N8-CIR-Bede/documentation>`__ where we welcome you to add issues
or to contribute by following the instructions in the README file. 


.. _faq-helpsupport:

How can I get further help and support?
---------------------------------------
Each institution has Research Software Engineer support for Bede, and you can
find the support email address for your institution `on the N8CIR website
<https://n8cir.org.uk/supporting-research/facilities/bede/rse-support-bede/>`__.
There is also a `slack workspace <https://n8cirbede.slack.com>`__ that you can join to get further support and
contact the Bede user community. To request access, please e-mail: marion.weinzierl@durham.ac.uk.

.. _faq-architecture-specific-eng:

How do I specialise my bash environment for Power 9 and Grace-Hopper systems?
-----------------------------------------------------------------------------

If you have modified your Bede user environment (``.bashrc``, or ``.bash_profile``) to make software available by default (i.e. conda),
you may need to modify your environment to set environment variables or source scripts based on the CPU architecture.

You can check the CPU architecture in bash using the ``uname`` command.

This allows you to set different environment variables on the ``aarch64`` Grace-Hopper nodes than on the ``ppc64le`` Power 9 nodes:

.. code-block:: bash

   # Get the CPU architecture
   arch=$(uname -i)
   if [[ $arch == "aarch64" ]]; then
      # Set variables and source scripts for aarch64
       export MYVAR=FOO
   elif [[ $arch == "ppc64le" ]]; then
      # Set variables and source scripts for ppc64le
       export MYVAR=BAR
   fi