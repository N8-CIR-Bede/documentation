.. _bede_connecting:

Connecting to the Bede HPC system
#################################

Connecting to Bede with SSH
===========================

The most versatile way to **run commands and submit jobs** on one of the clusters is to 
use a mechanism called `SSH <https://en.wikipedia.org/wiki/Secure_Shell>`__, 
which is a common way of remotely logging in to computers 
running the Linux operating system.  

To connect to another machine using SSH you need to 
have a SSH *client* program installed on your machine.  
macOS and Linux come with a command-line (text-only) SSH client pre-installed.  
On Windows there are various graphical SSH clients you can use, 
including *MobaXTerm*.


SSH client software on Windows
------------------------------

Download and install the *Installer edition* of `MobaXterm <https://mobaxterm.mobatek.net/download-home-edition.html>`_.

After starting MobaXterm you should see something like this:

.. image:: /images/mobaxterm-welcome.png
   :width: 50%
   :align: center

Click *Start local terminal* and if you see something like the following then please continue to :ref:`ssh`.

.. image:: /images/mobaxterm-terminal.png
   :width: 50%
   :align: center

SSH client software on Mac OS/X and Linux
-----------------------------------------

Linux and macOS (OS X) both typically come with a command-line SSH client pre-installed.


Establishing a SSH connection
-----------------------------

Once you have a terminal open run the following command to 
log in to a cluster: 

.. code-block:: bash

    ssh $USER@login1.bede.dur.ac.uk

    # Alternatively you can use the login node 2
    ssh $USER@login2.bede.dur.ac.uk

Here you need to:

* replace ``$USER`` with your username (e.g. ``te1st``)

You will then be asked for to enter your password. If the password is correct you 
should get a prompt resembling the one below: ::

    (base) [te1st@login1 ~]$

.. note::

    When you login to a cluster you reach one of two login nodes. 
    You **should not** run applications on the login nodes.
    Running ``srun`` gives you an interactive terminal 
    on one of the many worker nodes in the cluster.




Transferring files
==================

Transferring files with MobaXTerm (Windows)
-------------------------------------------

After connecting to Bede with MobaXTerm, you will see a files panel of the left of the screen. You can drag files from Windows explorer into the panel to upload the file 
or right clicking on the files in the panel and select ``Download`` to download the files to your machine.

Transferring files to/from Bede with SCP (Linux and Mac OS)
-----------------------------------------------------------

Secure copy (scp) can be used to transfer files between systems through the SSH protocol. 

To transfer from your machine to Bede (assuming our username is te1st):

.. code-block:: bash

    # Copy myfile.txt from the current directry to your Bede home directory
    scp myfile.txt te1st@login1.bede.dur.ac.uk:~/


To transfer from Bede to your machine:

.. code-block:: bash

    # Copy myfile.txt from the Bede home directory to current local directory
    scp te1st@login1.bede.dur.ac.uk:~/myfile.txt ./

