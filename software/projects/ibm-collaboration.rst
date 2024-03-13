.. _software-projects-ibm-collaboration:

IBM Collaboration
=================

.. |arch_availabilty_name| replace:: The "IBM Collaboration" Software environment
.. include:: /common/ppc64le-only.rst


On Bede, the ``ibm-collaboration`` project provides several software packages which were produced in collaboration with the system vendor `IBM <https://www.ibm.com/>`__.

* :ref:`Cryo-EM <software-environments-cryoem>` - a collection of software packages for life sciences including:
  
  * `RELION <https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Main_Page>`__
  * `CTFfind4 <https://grigoriefflab.umassmed.edu/ctffind4>`__
  * `MotionCor2 <https://emcore.ucsf.edu/ucsf-software>`__
  * `crYOLO <https://cryolo.readthedocs.io/en/stable/>`__
  * `ResMap <http://resmap.sourceforge.net/>`__

* :ref:`EMAN2 <software-applications-eman2>` - a scientific image processing suite with a primary focus on processing data from transmission electron microscopes.

For instructions on how to use these projects please see the :ref:`Cryo-EM <software-environments-cryoem>` and :ref:`EMAN2 <software-applications-eman2>` pages.

.. warning::

    The ``ibm-collaboration`` module does currently provide valid Environment Modules to load the included software packages. This will be addressed in the future.
    
    Instead, follow the instructions listed for the included projects to load the software packages via conda.