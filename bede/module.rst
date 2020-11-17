.. _bede_module:

Activating software using Environment Modules
=============================================

Overview and rationale
----------------------

'Environment Modules' are the mechanism by which much of the software is made available to the users of Bede.

To make a particular piece of software available a user will *load* a module e.g. you can load a particular version of the '``CUDA``' library with: ::

    module load cuda/10.2

This command manipulates `environment variables <https://en.wikipedia.org/wiki/Environment_variable>`_ to make this piece of software available.  
If you then want to switch to using a different version of ``CUDA`` (should another be installed on the cluster you are using) then you can run: ::

    module unload cuda/10.2
    
then load the other.  

You may wonder why modules are necessary: why not just install packages provided by the vender of the operating system installed on the cluster?
In shared high-performance computing environments such as Bede:

* users typically want control over the version of applications that is used (e.g. to give greater confidence that results of numerical simulations can be reproduced);
* users may want to use applications built using compiler X rather than compiler Y as compiler X might generate faster code and/or more accurate numerical results in certain situations;
* users may want a version of an application built with support for particular parallelisation mechanisms such as MPI for distributing work between machines, OpenMP for distributing work between CPU cores or CUDA for parallelisation on GPUs);
* users may want an application built with support for a particular library.

There is therefore a need to maintain multiple versions of the same applications on Bede.
Module files allow users to select and use the versions they need for their research.

If you switch to using a cluster other than Bede then you will likely find that environment modules are used there too.  
Modules are not the only way of managing software on clusters: increasingly common approaches include:

* the :ref:`Conda <bede_python_anaconda>` package manager (Python-centric but can manage software written in any language);

Basic guide
-----------

You can list all (loaded and unloaded) modules on Bede using: ::

    module avail

You can then load a module using e.g.: ::

    module load cuda/10.2

You can then load further modules e.g.::

    module load gcc/openmpi-3.0.3

Confirm which modules you have loaded using: ::

   module list

If you want to stop using a module (by undoing the changes that loading that module made to your environment): ::

    module unload cuda/10.2

or to unload all loaded modules: ::

    module purge

You can search for a module using: ::

    module avail |& grep -i somename


Some other things to be aware of:

* You can load and unload modules in both interactive and batch jobs;
* Modules may themselves load other modules.  If this is the case for a given module then it is typically noted in our documentation for the corresponding software;
* The order in which you load modules may be significant (e.g. if module A sets ``SOME_ENV_VAR=apple`` and module B sets ``SOME_ENV_VAR=pear``);
* Some related module files have been set up so that they are mutually exclusive e.g. on Bede the modules ``cuda/10.2`` and ``cuda/10.1`` cannot be loaded simultaneously (as users should never want to have both loaded).


Module Command Reference
------------------------
Here is a list of the most useful ``module`` commands. For full details, type ``man module`` at the command prompt on one of the clusters.

* ``module list`` – lists currently loaded modules
* ``module avail`` – lists all available modules
* ``module load modulename`` – loads module ``modulename``
* ``module unload modulename`` – unloads module ``modulename``
* ``module switch oldmodulename newmodulename`` – switches between two modules
* ``module show modulename`` - Shows how loading ``modulename`` will affect your environment
* ``module purge`` – unload all modules
* ``module help modulename`` – may show longer description of the module if present in the modulefile
* ``man module`` – detailed explanation of the above commands and others

More information on the Environment Modules software can be found on the `project's site <http://modules.sourceforge.net/>`_.