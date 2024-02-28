.. _software-environments-cryoem:

Cryo-EM Software Environment
============================

Documentation on the the Cryo-EM Software Environment for Life Sciences is available :download:`here <Cryo-EM_Bede.pdf>`. 
Note that this document is mainly based on the installation on `Satori <https://mit-satori.github.io>`_ and might have some inconsistencies with the Bede installation.

The Cryo-EM software package is provided by the :ref:`IBM Collaboration project <software-projects-ibm-collaboration>`.

It is a conda environment which provides the following software packages:

* `RELION <https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Main_Page>`__
* `CTFfind4 <https://grigoriefflab.umassmed.edu/ctffind4>`__
* `MotionCor2 <https://emcore.ucsf.edu/ucsf-software>`__
* `crYOLO <https://cryolo.readthedocs.io/en/stable/>`__
* `ResMap <http://resmap.sourceforge.net/>`__

To access these packages, first you must have a local conda installation set up and activated. 
See :ref:`Conda <software-applications-conda>` for instructions on how to install and enable conda.

The CyroEM conda environment can then be loaded using:

.. code-block:: bash

   conda activate /projects/bddir04/ibm-lfsapp/CryoEM

Once loaded, the included software applications can then be used.
