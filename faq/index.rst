FAQ
=====

This page contains answers to the questions most frequently asked by Bede
users. The question list will be updated over time as more questions are
asked - if there is anything that you think should definitely be answered
here, please let us know at <email.address@dur.ac.uk>.

How long can I run jobs for?
----------------------------

You can run jobs for a maximum of 48 hours, in either the `gpu` or `infer`
partitions.

How can I acknowledge Bede in published or presented work?
----------------------------------------------------------------

You can acknowledge Bede using the standard text that we provide. You can
find this `here <https://bede-documentation.readthedocs.io/en/latest/usage/index.html#acknowledging-bede>`__.

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
about the Bede filesystems `here <https://bede-documentation.readthedocs.io/en/latest/usage/index.html#file-storage>`__.

Where should I put project data?
--------------------------------

Project data that does not need to be backed up, e.g. intermediate results,
should be stored in the :code:`/nobackup/projects/<project>` directory, where
:code:`<project>` is your Bede project code.

Project data that should be backed up, e.g. final results that would be painful
to recompute, should be store in the :code:`/projects/<project>` directory.

.. note::
  Backups are **NOT** currently implemented on the :code:`/projects` filesystem!
