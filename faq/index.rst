FAQ
=====

This page contains answers to the questions most frequently asked by Bede
users. 
The question list will be updated over time as more questions are
asked - if there is anything that you think should definitely be answered
here, please let us know by `Opening an Issue on GitHub <https://github.com/N8-CIR-Bede/documentation/issues/new>`__ or by contacting your `local Bede RSE <https://n8cir.org.uk/supporting-research/facilities/bede/rse-support-bede/>`__ .

How long can I run jobs for?
----------------------------

You can run jobs for a maximum of 48 hours, in either the `gpu` or `infer`
partitions.

How can I acknowledge Bede in published or presented work?
----------------------------------------------------------------

You can acknowledge Bede using the standard text that we provide. You can
find this :ref:`here <usage-acknowledging-bede>`.

How can I check my home directory quota?
----------------------------------------

You can use the following command:

.. code-block:: text

  quota -s

  Disk quotas for user klcm500 (uid 639800132): 
       Filesystem   space   quota   limit   grace   files   quota   limit   grace
  nfs.bede.dur.ac.uk:/nfs/users
                   79544K      0K  20480M            1071       0       0

This tells me that, in this case, I have a limit of :code:`20480M`, and am 
currently using :code:`79544K` across 1071 files. You can find more information
about the Bede filesystems :ref:`here <usage-file-storage>`.

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

The 'Using Bede' page runs through how to get registered, how to log in to the
machine and how to run jobs, a link to this page can be found :ref:`here
<using-bede>`.

How can I add my own software?
------------------------------

It is recommended that you use spack to extend the installed software on the
system, there are instructions on how to do this :ref:`here <software-environments>`,
along with further information about alternatives.


Is MATLAB available on Bede?
----------------------------

Unfortunately MATLAB is not available on Bede. MATLAB is not currently supported on IBM systems with a Power architecture.
It is possible to install Octave on Bede which can work as an alternative, although there is
currently no official support for this.


How can I suggest improvements or contribute to this documentation?
-------------------------------------------------------------------

The Bede documentation is maintained on github at
https://github.com/N8-CIR-Bede/documentation where we welcome you to add issues
or to contribute by following the instructions in the README file. 


How can I get further help and support?
---------------------------------------
Each institution has Research Software Engineer support for Bede, and you can
find the support email address for your institution `here
<https://n8cir.org.uk/supporting-research/facilities/bede/rse-support-bede/>`__.
There is also a slack workspace that you can join to get further support and
contact the Bede user community. To request access, please e-mail: marion.weinzierl@durham.ac.uk.

